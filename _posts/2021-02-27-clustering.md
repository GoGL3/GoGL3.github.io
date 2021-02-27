---
title: "[NLP] Code - BERT + KMeans"
date: 2021-02-27 16:000 -0400
author : 조경민
categories :
  - BERT
  - NLP
  - KMeans
  - tSNE
  - UMAP
---

```ruby
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import umap
```

### Import dataset (Keyword sets)
```ruby
keyword = pd.read_csv('keyword.csv')
def clean_alt_list(list_):
    list_ = list_.replace('[', '')
    list_ = list_.replace(']', '')
    list_ = list_.replace("'","")
    list_ = list_.replace(',','')
    return list_
keyword.keyword_list = keyword.keyword_list.apply(clean_alt_list)
lst_keyword = [x for x in keyword.keyword_list]
```

![Alt text](/assets/df1.jpg)
![Alt text](/assets/df2.jpg)  

### Extract Embedding values per keyword set using BERT
```ruby
import torch
from transformers import BertTokenizer, BertModel, DistilBertModel
from tokenization_kobert import KoBertTokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

class Embed_Kor:
    def __init__(self, sent, pretrain_ver='monologg/kobert'):
        self.text = sent
        self.ver = pretrain_ver

    def tokenization_kor(self):
        marked_text = '[CLS]' + self.text + '[SEP]'
        tokenizer = KoBertTokenizer.from_pretrained(self.ver)
        tokenized_text = tokenizer.tokenize(marked_text)
        input_ids = tokenizer.convert_tokens_to_ids(tokenized_text)
        att_mask = tokenizer.get_special_tokens_mask(input_ids[1:-1])
        type_ids = tokenizer.create_token_type_ids_from_sequences(input_ids[1:-1])
        return input_ids, att_mask, type_ids

    def _transformer_kor(self):
        model, vocab = get_pytorch_kobert_model()
        input_ids, att_mask, type_ids = self.tokenization_kor()
        input_ids = torch.LongTensor([input_ids])
        att_mask = torch.LongTensor([att_mask])
        type_ids = torch.LongTensor([type_ids])
        sequence_output, pooled_output = model(input_ids, att_mask, type_ids)
        final_embed = sequence_output[0]
        return final_embed

    if __name__=="__main__":
        print('Transformer-Korean ver. ready')


class Embed_multi:
    def __init__(self, sent, pretrain_ver='bert-base-multilingual-uncased', unit='sentence'):
        self.text = sent
        self.ver = pretrain_ver
        self.unit = unit

    def tokenization_multi(self):
        tokenizer = BertTokenizer.from_pretrained(self.ver)
        marked_text = '[CLS]' + self.text + '[SEP]'
        tokenized_text = tokenizer.tokenize(marked_text)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        segment_ids = [1] * len(tokenized_text)
        return indexed_tokens, segment_ids

    def _transformer_multi(self):
        model = BertModel.from_pretrained(self.ver, output_hidden_states = True)
        indexed_tokens, segment_ids = self.tokenization_multi()
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segment_ids])
        with torch.no_grad():
            outputs = model(tokens_tensor, segments_tensors)
        hidden_states = outputs[2]
    
    if self.unit == 'word':
        ## Stack all hidden states (outputs from 13 layers)
        token_embeddings = torch.stack(hidden_states, dim=0)
        ## Squeeze the batch dimension (since we only deal with ONE sentence -> batch=1)
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        ## Swap dimensions 0 and 1 ([layer,token,features=768] -> [token,layer,features=768])
        token_embeddings = token_embeddings.permute(1,0,2)
        ## Summing last 4 layers
        token_vecs_sum = []
        for token in token_embeddings:
            sum_vec = torch.sum((token[-4:]), dim=0)
            token_vecs_sum.append(sum_vec)
        return token_vecs_sum

    elif self.unit == 'sentence':
        token_vecs = hidden_states[-2][0]
        sent_embed = torch.mean(token_vecs, dim=0)
        return sent_embed

    if __name__=="__main__":
        print('Transformer-Multilingual ver. ready')
```
```ruby
import pickle

embed_keyword = []
for idx, sent in enumerate(sents):
    module = Embed_multi(sent)
    res = module._transformer_multi()
    res = res.numpy()
    embed_keyword.append(res)

    if (idx % 100) == 0:
        with open('keyword_embed.pkl', 'wb') as f:
            pickle.dump(embed_keyword, f)
        print('Total {} sentences have been embedded.. Please wait a bit more.. :('.format(idx+1))
```
_(** CAUTION: Depending on the size of data, it can take LOOOONG time !!)_  

### Import toy data obtained by above method
```ruby
with open('keyword_embed.pkl', 'rb') as f:
    x = pickle.load(f)
data = np.asarray(x)
```  

### Dimension Reduction - 1) UMAP
```ruby
reducer = umap.UMAP()
reduced_umap = reducer.fit_transform(data)
reduced_umap[:5,:]
```

![Alt text](/assets/df3.jpg)

```ruby
plt.scatter(
    reduced_umap[:, 0],
    reduced_umap[:, 1])
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP projection result', fontsize=24)
```

![Alt text](/assets/umap.jpg)  

### Dimension Reduction - 2) tSNE
```ruby
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, metric='cosine')
reduced_tsne = tsne.fit_transform(data)

plt.scatter(
    reduced_tsne[:, 0],
    reduced_tsne[:, 1])
plt.gca().set_aspect('equal', 'datalim')
plt.title('tSNE projection result', fontsize=24)
```

![Alt text](/assets/tsne.jpg)  

### KMeans
#### 1) Full data version
```ruby
from sklearn.cluster import KMeans

df = pd.DataFrame(data)
df['keyword_list'] = lst_keyword[:1501]

model = KMeans(n_clusters=30)
model.fit(data)
predict = pd.DataFrame(model.predict(data))
predict.columns = ['predict']
r = pd.concat([df,predict], axis=1)

cluster_keyword = r.groupby('predict')['keyword_list'].apply(lambda x: ' '.join(x))
pd.DataFrame(cluster_keyword)
```  


#### 2-1) Reduced data version - UMAP
```ruby
model_umap = KMeans(n_clusters=30)
model_umap.fit(reduced_umap)
predict_umap = pd.DataFrame(model_umap.predict(reduced_umap))
predict_umap.columns = ['predict']

df_reduced_umap = pd.DataFrame(reduced_umap)
df_reduced_umap['keyword_list'] = lst_keyword[:1501]
r2 = pd.concat([df_reduced_umap,predict_umap], axis=1)

cluster_keyword_umap = r2.groupby('predict')['keyword_list'].apply(lambda x: ' '.join(x))
pd.DataFrame(cluster_keyword_umap)
```  

#### 2-2) Reduced data version - tSNE
```ruby
model_tsne = KMeans(n_clusters=30)
model_tsne.fit(reduced_tsne)
predict_tsne = pd.DataFrame(model_tsne.predict(reduced_tsne))
predict_tsne.columns = ['predict']

df_reduced_tsne = pd.DataFrame(reduced_tsne)
df_reduced_tsne['keyword_list'] = lst_keyword[:1501]
r3 = pd.concat([df_reduced_tsne,predict_tsne], axis=1)

cluster_keyword_tsne = r3.groupby('predict')['keyword_list'].apply(lambda x: ' '.join(x))
pd.DataFrame(cluster_keyword_tsne)
```
