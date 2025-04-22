# Main Code - Fixing demasking of comments :- Main Code
import re
import os
import gzip
import time
import json
import spacy
import string
import random
import tempfile
import pandas as pd
from faker import Faker
from pathlib import Path
from collections import defaultdict, deque
from presidio_analyzer import AnalyzerEngine

# Variable defined
ID, fake_data, used_urls, entity_columns  = {}, {}, set(), {}
COMMENT_KEYWORDS = ['comments', 'description', 'remarks', 'note', 'feedback', 'observation']

url_extensions = [
    ".com", ".net", ".org", ".edu", ".gov", ".co", ".us", ".uk", ".in", ".ru",
    ".jp", ".cn", ".de", ".fr", ".it", ".nl", ".es", ".br", ".au", ".ca",
    ".ch", ".se", ".no", ".za", ".mx", ".ar", ".be", ".kr", ".pl", ".tr",
    ".ua", ".ir", ".sa", ".ae", ".my", ".sg", ".hk", ".tw", ".nz", ".id",
    ".th", ".ph", ".vn", ".bd", ".lk", ".np", ".pk", ".cz", ".gr", ".hu",
    ".fi", ".dk", ".il", ".ie", ".pt", ".sk", ".si", ".ro", ".bg", ".rs",
    ".lt", ".lv", ".ee", ".hr", ".ba", ".md", ".ge", ".kz", ".by", ".tm",
    ".uz", ".af", ".qa", ".om", ".kw", ".bh", ".ye", ".jo", ".lb", ".sy",
    ".iq", ".ps", ".az", ".am", ".kg", ".mn", ".bt", ".mv", ".mm", ".kh",
    ".la", ".tl", ".sb", ".fj", ".pg", ".to", ".tv", ".ws", ".fm", ".ki"
]
with open("test.json", "r", ) as f: fake_data_list = json.load(f)
# Objects
nlp = spacy.load("en_core_web_sm")
fake=Faker()
analyzer = AnalyzerEngine()

for data in fake_data_list:
    for key,value in data.items(): fake_data[key]=deque(value)
domain_pool = list(fake_data.get('url', deque()))

entity_mapping={
    'names':'PERSON',
    'emails':'EMAIL_ADDRESS',
    'phone':'PHONE_NUMBER',
    'location':'LOCATION',
    'credit':'CREDIT_CARD',
    'url':'URL',
    'country':'COUNTRY',
    'company':"ORG",
    'id':'ID',
}

mapping_file="mapping.json"
forward_mapping=defaultdict(dict)
reverse_mapping=defaultdict(dict)
comment_entity_positions = defaultdict(dict)

if os.path.exists(mapping_file):
    with open(mapping_file, "r") as f:
        mapping_data = json.load(f)
        forward_mapping.update(mapping_data.get("forward_mapping", {}))
        reverse_mapping.update(mapping_data.get("reverse_mapping", {}))

# To track time
def time_it(func):
    """Decorator to measure execution time of functions."""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f'\n⏳ Execution time {func.__name__}: {end-start:.6f} seconds')
        return result
    return wrapper

# For comments logic
def get_comment_columns(df):
    return [col for col in df.columns if any(kw in col.lower() for kw in COMMENT_KEYWORDS)]
@time_it
def mask_comment_columns(df, mapping):
    comment_cols = get_comment_columns(df)
    print('Inside mask comment cols:', comment_cols)

    for col in comment_cols:
        for i, text in df[col].astype(str).items():
            masked_text, entities = mask_comment_text(text, mapping)
            df.at[i, col] = masked_text
            if entities: comment_entity_positions[col][i] = entities
    return df

def mask_comment_text(text, mapping):
    if not text.strip(): return text, []

    doc = nlp(text)
    masked_text = text
    entities = []
    offset = 0

    for token in doc:
        if token.pos_ == "PROPN" or token.ent_type_:
            original = token.text
            fake = mapping['names'].get(original, None)

            if fake and fake != original:
                pattern = re.compile(rf"\b{re.escape(original)}(?=[\W_]|$)")
                match = pattern.search(masked_text, token.idx + offset)
                if match:
                    start, end = match.span()
                    masked_text = masked_text[:start] + fake + masked_text[end:]
                    entities.append({
                        "start": start,
                        "end": start + len(fake),
                        "original": original,
                        "fake": fake
                    })
                    offset += len(fake) - len(original)
    return masked_text, entities

@time_it
def unmask_comment_columns(df, reverse_mapping, entity_metadata):
    comment_cols = get_comment_columns(df)
    for col in comment_cols:
        if col not in entity_metadata: continue

        for i, text in df[col].astype(str).items():
            if i in entity_metadata[col]:
                df.at[i, col] = unmask_comment_text(text, entity_metadata[col][i], reverse_mapping)
    return df

def unmask_comment_text(text, entities, reverse_mapping):
    for ent in sorted(entities, key=lambda x: -x['start']):
        fake = ent['fake']
        original = reverse_mapping['names'].get(fake, ent['original'])
        text = text[:ent['start']] + original + text[ent['end']:] 
    return text

@time_it
def analyze_column(df):
    # Step 1: Use Presidio to analyze all columns
    comment_cols = get_comment_columns(df)

    for col in df.columns:
        if col in comment_cols: continue
        if 'id' in col.lower():
            entity_columns[col] = 'ID'
        elif 'country' in col.lower():
            entity_columns[col] = 'COUNTRY'
        else:
            unique_values = df[col].dropna().astype(str).unique()[:25]
            entity_counts = {}

            for value in unique_values:
                results = analyzer.analyze(text=value, language='en')
                for result in results:
                    entity_counts[result.entity_type] = entity_counts.get(result.entity_type, 0) + 1

            if entity_counts:
                predominant_entity = max(entity_counts, key=entity_counts.get)
                entity_columns[col] = predominant_entity
                if predominant_entity=="LOCATION":
                  org_count=0
                  for value in unique_values:
                    doc=nlp(value)
                    for ent in doc.ents:
                      if ent.label_=="ORG":
                        org_count+=1
                  if org_count>5:
                    predominant_entity="ORG"
                entity_columns[col]=predominant_entity

    # Step 2: Use SpaCy to analyze non-numeric and unclassified columns
    for col in df.select_dtypes(exclude=['number']).columns:
        if col not in entity_columns:
            unique_values = df[col].dropna().astype(str).unique()[:25]
            org_count = 0

            for value in unique_values:
                doc = nlp(value)
                for ent in doc.ents:
                    if ent.label_ == 'ORG':
                        org_count += 1

            # If more than half of the sample values are ORG, classify as ORG
            if org_count > 12:
                entity_columns[col] = 'ORG'

    return entity_columns

def modify_fake_value(category, base_fake_value, counter):
    if category == "names": return f"{base_fake_value}{string.ascii_lowercase[counter % 26]}"
    elif category == "emails":
        name, domain = base_fake_value.split("@")
        return f"{name}{counter}@{domain}"
    elif category in {"location", "country"}: return f"{base_fake_value}, District {counter % 100_000_000 + 1}"
    elif category == "url":
        # Check if the URL already exists, if so, append a country extension
        fake_value = base_fake_value
        while fake_value in used_urls:
            ext = random.choice(url_extensions)
            if not fake_value.endswith(ext):
                fake_value += ext
        used_urls.add(fake_value)
        return fake_value
    elif category == "phone": return f"{base_fake_value[:-2]}{counter % 100:02d}"
    elif category == "company": return f"{base_fake_value} Group {counter % 100_000_000 + 1}"
    elif category == "credit": return f"{base_fake_value[:-4]}{counter % 10000:04d}"
    else: return f"{base_fake_value}-{counter}" 

 
def get_fake_value(category, original_value):
    global ID
    fake_value = None
    if category == "names":
        original_tokens = str(original_value).split()
        fake_tokens = []
        used_fake_names = set(reverse_mapping[category])

        for token in original_tokens:
            if token in forward_mapping[category]:
                fake_token = forward_mapping[category][token]
            else:
                fake_token = None

                if fake_data.get(category):
                    for _ in range(len(fake_data[category])):
                        candidate = fake_data[category].popleft()
                        fake_data[category].append(candidate)

                        if candidate not in used_fake_names and candidate != token:
                            fake_token = candidate
                            break
                        else:
                            # Modify it if already used
                            counter = 0
                            base = candidate
                            while True:
                                modified = modify_fake_value(category, base, counter)
                                if modified not in used_fake_names and modified != token:
                                    fake_token = modified
                                    break
                                counter += 1
                            break

                # Absolute fallback: no fake_data or nothing worked
                if not fake_token:
                    counter = 0
                    base_pool = list(used_fake_names)
                    random.shuffle(base_pool)

                    for base in base_pool[:10]:  # Try 10
                        fake_token = modify_fake_value(category, base, counter)
                        if fake_token not in used_fake_names and fake_token != token:
                            break
                        counter += 1
                    else:
                        # Final fallback: just keep modifying the original token
                        while True:
                            fake_token = modify_fake_value(category, token, counter)
                            if fake_token not in used_fake_names:
                                break
                            counter += 1

                # Save mappings
                forward_mapping[category][token] = fake_token
                reverse_mapping[category][fake_token] = token
                used_fake_names.add(fake_token)

            fake_tokens.append(fake_token)

        return " ".join(fake_tokens)

    if original_value in forward_mapping[category]: return forward_mapping[category][original_value]
    # Special case for ID
    if category == 'id':
        length = 6
        while True:
            fake_value = fake.bothify(text=f'ID-{"#"*length}')
            if fake_value not in ID:
                ID[fake_value] = True
                break
            if len(ID) >= 10 ** length: length += 1
    elif category == 'url':
        domain1, domain2 = random.sample(domain_pool, 2)
        base_fake_value = f"https://{domain1.lower()}/{domain2.lower()}.co"

        if base_fake_value not in reverse_mapping["url"]:
            fake_value = base_fake_value
        else:
            counter = len(reverse_mapping["url"])
            fake_value = modify_fake_value("url", base_fake_value, counter)

    elif fake_data.get(category):
      length = len(fake_data[category])
      for _ in range(length):
            candidate = fake_data[category].popleft()
            fake_data[category].append(candidate)  # Reinsert at end (rotation)
            if candidate not in reverse_mapping[category]:
                fake_value = candidate
                break
      else:
          counter = len(reverse_mapping[category])
          base_fake_value = (
              random.choice(list(reverse_mapping[category]))
              if reverse_mapping[category]
              else f"{category}_"
          )
          fake_value = modify_fake_value(category, base_fake_value, counter)

    if not fake_value or fake_value in reverse_mapping[category]:
        counter = len(reverse_mapping[category])
        base = fake_value if fake_value else f"{category}_"
        fake_value = modify_fake_value(category, base, counter)

    forward_mapping[category][original_value] = fake_value
    reverse_mapping[category][fake_value] = original_value

    return fake_value


@time_it
def mask_dataframe(df):
    for col, entity in entity_columns.items():
        matching_keys = [
            key for key, value in entity_mapping.items() if value == entity
        ]
        if matching_keys:
            category = matching_keys[0]
            df[col] = df[col].astype(str).apply(
                lambda x: get_fake_value(category, x) if x else x
            )
    return df

def restore_original_value(category, fake_value):
    return reverse_mapping[category].get(fake_value, fake_value)

@time_it
def unmask_dataframe(df):
    for col, entity in entity_columns.items():
        matching_keys = [key for key, value in entity_mapping.items() if value == entity]
        if matching_keys:
            category = matching_keys[0]

            if category == "names":
                def reverse_name(val):
                    tokens = str(val).split()
                    original_tokens = [reverse_mapping[category].get(tok, tok) for tok in tokens]
                    return " ".join(original_tokens)
                df[col] = df[col].astype(str).apply(lambda x: reverse_name(x) if pd.notna(x) else x)

            else:
                df[col] = df[col].astype(str).apply(
                    lambda x: restore_original_value(category, str(x)) if pd.notna(str(x)) else x
                )
    df = unmask_comment_columns(df, reverse_mapping, comment_entity_positions)
    return df

@time_it
def compare_files(original_csv, restored_csv):
    """Compare two CSV files and return if they are identical."""
    original_df = pd.read_csv(original_csv, dtype=str).sort_index(axis=1).reset_index(drop=True)
    restored_df = pd.read_csv(restored_csv, dtype=str).sort_index(axis=1).reset_index(drop=True)

    is_identical = original_df.equals(restored_df)
    print(f"\n📊 Are files identical? {'✅ Yes' if is_identical else '❌ No'}")

    if not is_identical:
        print("⚠️ The restored file does not match the original. Investigate the mapping or masking logic.")

    return is_identical

def de_anonymize_paragraph(text):
  for category,mapping in reverse_mapping.items():
    for fake_value,original_value in mapping.items():
      if fake_value in text:
        text=text.replace(fake_value,original_value)
  return text

def save_mapping(filename):
    mapping_data={
        "filename":filename,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "forward_mapping":forward_mapping,
        "reverse_mapping":reverse_mapping
    }
    with open(mapping_file, "a") as f:
        json.dump(mapping_data, f, indent=4)

def process_file(input_file, file_ext):
    input_base = os.path.splitext(os.path.basename(input_file))[0]
    output_dir = os.path.join(".", input_base)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if file_ext == ".xlsx":
        xl = pd.read_excel(input_file, sheet_name=None, dtype=str, engine='openpyxl')
        
        for sheet_name, df in xl.items():
            print(f"\n🔍 Processing sheet: {sheet_name}")

            original_csv = os.path.join(output_dir, f"{sheet_name}_original.csv")
            masked_csv = os.path.join(output_dir, f"{sheet_name}_masked.csv")
            restored_csv = os.path.join(output_dir, f"{sheet_name}_restored.csv")

            df.to_csv(original_csv, index=False)

            entity_columns = analyze_column(df)
            print(f"Detected entities: {entity_columns}")

            masked_df = mask_dataframe(df.copy())
            masked_df = mask_comment_columns(masked_df, forward_mapping)
            restored_df = unmask_dataframe(masked_df.copy())

            masked_df.to_csv(masked_csv, index=False)
            restored_df.to_csv(restored_csv, index=False)

            compare_files(original_csv, restored_csv)

            save_mapping(f'{input_file}-->{sheet_name}')
            entity_columns.clear()


    elif file_ext == ".csv":
        df = pd.read_csv(input_file, dtype=str, low_memory=False)
        print(f"\n🔍 Processing CSV: {input_file}")

        original_csv = os.path.join(output_dir, "original.csv")
        masked_csv = os.path.join(output_dir, "anonymized.csv")
        restored_csv = os.path.join(output_dir, "restored.csv")

        if not os.path.exists(output_dir): os.makedirs(output_dir)

        df.to_csv(original_csv, index=False)

        entity_columns = analyze_column(df)
        print(f"Detected entities: {entity_columns}")

        masked_df = mask_dataframe(df.copy())
        masked_df = mask_comment_columns(masked_df, forward_mapping)
        restored_df = unmask_dataframe(masked_df.copy())

        masked_df.to_csv(masked_csv, index=False)
        restored_df.to_csv(restored_csv, index=False)

        compare_files(original_csv, restored_csv)

        save_mapping(input_file)
    else:
        raise ValueError("Unsupported file type. Use .csv or .xlsx only.")

if __name__ == "__main__":
    input_file = "for_colab_test.csv"
    file_ext = os.path.splitext(input_file)[-1].lower()
    process_file(input_file, file_ext)