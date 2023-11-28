import pandas as pd
from nltk.tokenize import word_tokenize
import openai
from preprocess_bert import *
openai.api_key = "" # add your key here.
def rejoin_sentence(words):
    sentence = ''
    for word in words:
        if word in [".", ",", "!", "?", ";", ":"]:
            sentence = sentence.rstrip() + word + ' '
        else:
            sentence += word + ' '
    return sentence.strip()
def chatGPT_generate(prompt):
    CoT_prompt = "Q:Please help me summarize the summary of the news abstraction, please focus on the categories in the news and keep that summary under 30 tokens in length:\n News abstraction:"
    messages = [
        {
            "role": "system",
            "content": "You are an AI assistant that helps people find information.",  # system prompt added
        }
    ]
    prompt = CoT_prompt + prompt
    prompt+"\nA:"
    message_prompt = {"role": "user", "content": prompt}
    messages.append(message_prompt)
    flag = True
    i = 0
    print(prompt)
    while flag:
        try:
            MODEL = "gpt-3.5-turbo"
            response = openai.ChatCompletion.create(
                model=MODEL,  # 35-turbo
                messages=messages,
                temperature=0.4,
                max_tokens=300,
                top_p=0.9,
                frequency_penalty=0,
                presence_penalty=0,
            )
            result = response["choices"][0]["message"]["content"]
            flag = False
        except:
            i += 1
            if i == 10:
                exit(-1)

    print(result)

    return result

data_root_path = '/home/tanglujay/LLM-Hirec/output/'
embedding_path = '/home/tanglujay/LLM-Hirec/'
KG_root_path = '/home/tanglujay/LLM-Hirec/data'
range_start = 0
range_end = 25630
sample_size = 500

# 从 range_start 到 range_end 中随机选择 sample_size 个不同的数字
random_sample = random.sample(range(range_start, range_end + 1), sample_size)
print(random_sample)
df = pd.read_csv(data_root_path+'docs.tsv', sep='\t',header=None)
def modify_abs(x):
    x = x.rejoin_sentence
    return x.lower()
word_count = 0
from tqdm import tqdm
for index, row in tqdm(df.iterrows()):
    # 修改第三列的值（列索引为 2）
    # 这里写你的修改逻辑，例如将值增加 1
    # print(row[4])
    
    if type(row[4]) != str:
        continue
    if len(row[4].split(" ")) <=30:
        continue
    word_count+=1
    if word_count in random_sample:
        print(word_count)
        # print(row[4])
        # print(chatGPT_generate(row[4]))
    # print(row[4])
        result = chatGPT_generate(row[4])
        # print(result)
        df.at[index, 4] = result

print(word_count)

# 保存修改后的 DataFrame 到新的 TSV 文件
df.to_csv('modified_example.tsv', sep='\t', index=False)

# TODO: abstract
# news,news_index,category_dict,subcategory_dict,word_dict = read_news()
