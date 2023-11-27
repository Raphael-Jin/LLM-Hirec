# %%
import os

# %%
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# %%
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
 
config = tf.ConfigProto()  
config.gpu_options.allow_growth=True  
session = tf.Session(config=config)
 
KTF.set_session(session)

# %%
from preprocessing import *
from models import *
from hypers import *
from utils import *

# %%


# %%
data_root_path = '../data/output/'
embedding_path = '../data/embedding/'
KG_root_path = '../data/Opendata_KG/'

# %%
news,news_index,category_dict,subcategory_dict,word_dict = read_news(data_root_path,'docs.tsv')
news_title,news_vert,news_subvert=get_doc_input(news,news_index,category_dict,subcategory_dict,word_dict)
news_entity,news_entity_np,EntityId2Index = load_news_entity(news_index,KG_root_path)
news_info = np.concatenate([news_title,news_entity_np],axis=-1)

# %%
train_session = read_clickhistory(news_index,data_root_path,'train.tsv')
train_user = parse_user(news_index,train_session)
train_sess, train_user_id, train_label = get_train_input(news_index,train_session)

# %%
test_session = read_clickhistory(news_index,data_root_path,'test.tsv')
test_user = parse_user(news_index,test_session)
test_impressions, test_userids = get_test_input(news_index,test_session)

# %%
title_word_embedding_matrix, have_word = load_matrix(embedding_path,word_dict)
entity_emb_matrix = load_entity_embedding(KG_root_path,EntityId2Index)

# %%
index2nid = {}
for nid, nix in news_index.items():
    index2nid[nix] = nid

# %%
vert_subvert_mask_table = np.zeros((1,len(category_dict),len(subcategory_dict)))
for nid in range(1,len(news_vert)):
    v = news_vert[nid]-1
    sv = news_subvert[nid]-1
    vert_subvert_mask_table[0,v,sv] = 1

# %%


# %%
from keras.utils import Sequence

class get_hir_train_generator(Sequence):
    def __init__(self,mask_prob,news_scoring,index2nid,news_vert, subvert,news_entity, news_entity_id, clicked_news,user_id, news_id, label, batch_size, decay_rate = 0.9):
        self.news_emb = news_scoring
        self.vert = news_vert
        self.subvert = subvert
        self.entity = news_entity
        self.entity_id = news_entity_id
        self.index2nid = index2nid
        
        self.clicked_news = clicked_news

        self.user_id = user_id
        self.doc_id = news_id
        self.label = label
        
        self.mask_prob = mask_prob
        
        self.batch_size = batch_size
        self.ImpNum = self.label.shape[0]
        # self.clicked_news = self.apply_decay(clicked_news, decay_rate)

    @staticmethod
    def apply_decay(clicked_news, decay_rate):
        # Assuming clicked_news is a list of tuples [(news_id, timestamp), ...]
        max_timestamp = max(t for _, t in clicked_news)
        decayed_clicked_news = []
        for news_id, timestamp in clicked_news:
            time_diff = max_timestamp - timestamp
            decay_factor = decay_rate ** time_diff
            decayed_clicked_news.append((news_id, decay_factor))
        return decayed_clicked_news


    def __len__(self):
        return int(np.ceil(self.ImpNum / float(self.batch_size)))
    
    def __get_news(self,docids):
        news_emb = self.news_emb[docids]
        vert = self.vert[docids]
        subvert = self.subvert[docids]
        entity = self.entity[docids]
        return news_emb, vert, subvert, entity
        

    def __getitem__(self, idx):
        start = idx*self.batch_size
        ed = (idx+1)*self.batch_size
        if ed> self.ImpNum:
            ed = self.ImpNum
            
        label = self.label[start:ed]

        doc_ids = self.doc_id[start:ed]
        title, vert, subvert, entity = self.__get_news(doc_ids)
        
        user_ids = self.user_id[start:ed]
        clicked_ids = self.clicked_news[user_ids]
        user_title, user_vert, user_subvert, user_entity = self.__get_news(clicked_ids)
        
        vert_subvert_mask_input = np.zeros((len(user_subvert),len(category_dict),len(subcategory_dict),))
        for bid in range(len(user_subvert)):
            for nid in range(len(user_subvert[bid])):
                sv = user_subvert[bid][nid]
                if sv ==0:
                    continue
                sv -= 1
                vert_subvert_mask_input[bid,:,sv] = 1
        vert_subvert_mask_input = vert_subvert_mask_input*vert_subvert_mask_table

        
        
        user_vert = keras.utils.to_categorical(user_vert,len(category_dict)+1)
        user_vert = user_vert.transpose((0,2,1))
        user_vert = user_vert[:,1:,:]
        user_vert_mask = user_vert.sum(axis=-1)
        
        vert = keras.utils.to_categorical(vert,len(category_dict)+1)
        vert = vert[:,:,1:]
        
        user_subvert = keras.utils.to_categorical(user_subvert,len(subcategory_dict)+1)
        user_subvert = user_subvert.transpose((0,2,1))
        user_subvert = user_subvert[:,1:,:]
        user_subvert_mask = user_subvert.sum(axis=-1)
                
        subvert = keras.utils.to_categorical(subvert,len(subcategory_dict)+1)
        subvert = subvert[:,:,1:]
    
        user_vert_num = np.array(user_vert.sum(axis=-1),dtype='int32')
        user_subvert_num = np.array(user_subvert.sum(axis=-1),dtype='int32')

        user_subvert_mask = np.array(user_subvert_mask>0,dtype='float32')
        user_vert_mask = np.array(user_vert_mask>0,dtype='float32')
        vert_subvert_mask_input = np.array(vert_subvert_mask_input>0,dtype='float32')
        
        rw_vert = user_vert_num/(user_vert_num.sum(axis=-1).reshape((len(user_vert_num),1))+10**(-8)) #(bz,18)
        rw_subvert = user_subvert_num/(user_subvert_num.sum(axis=-1).reshape((len(user_subvert_num),1))+10**(-8)) #(bz,300)
        
        
        
        rw_vert = rw_vert.reshape((rw_vert.shape[0],1,rw_vert.shape[1]))
        rw_subvert = rw_subvert.reshape((rw_subvert.shape[0],1,rw_subvert.shape[1])) #(bz,1,18)
        
        rw_vert = (rw_vert*vert).sum(axis=-1)
        rw_subvert = (rw_subvert*subvert).sum(axis=-1)
        
        train_mask = np.random.uniform(0,1,size=(ed-start,1)) > self.mask_prob
        train_mask = np.array(train_mask,dtype='float32')
        
        rw_vert = rw_vert*train_mask
        rw_subvert = rw_subvert*train_mask



        return ([title,vert,subvert,user_title, user_vert,user_vert_mask,user_subvert,user_subvert_mask,vert_subvert_mask_input,user_vert_num,user_subvert_num,rw_vert,rw_subvert],[label])
    
    
class get_hir_user_generator(Sequence):
    def __init__(self,news_emb,news_vert,news_subvert,news_entity, clicked_news,batch_size):
        self.news_emb = news_emb
        self.vert = news_vert
        self.subvert = news_subvert
        self.entity = news_entity
        
        self.clicked_news = clicked_news

        self.batch_size = batch_size
        self.ImpNum = self.clicked_news.shape[0]
        
    def __len__(self):
        return int(np.ceil(self.ImpNum / float(self.batch_size)))

    
    def __get_news(self,docids):
        news_emb = self.news_emb[docids]
        vert = self.vert[docids]
        subvert = self.subvert[docids]
        entity = self.entity[docids]
        return news_emb, vert, subvert, entity
    
    
    def __getitem__(self, idx):
        start = idx*self.batch_size
        ed = (idx+1)*self.batch_size
        if ed> self.ImpNum:
            ed = self.ImpNum
            
        clicked_ids = self.clicked_news[start:ed]
        user_title, user_vert, user_subvert, user_entity = self.__get_news(clicked_ids)
        
        vert_subvert_mask_input = np.zeros((len(user_subvert),len(category_dict),len(subcategory_dict),))
        for bid in range(len(user_subvert)):
            for nid in range(len(user_subvert[bid])):
                sv = user_subvert[bid][nid]
                if sv ==0:
                    continue
                sv -= 1
                vert_subvert_mask_input[bid,:,sv] = 1
        vert_subvert_mask_input = vert_subvert_mask_input*vert_subvert_mask_table

        
        
        user_vert = keras.utils.to_categorical(user_vert,len(category_dict)+1)
        user_vert = user_vert.transpose((0,2,1))
        user_vert = user_vert[:,1:,:]
        user_vert_mask = user_vert.sum(axis=-1)
        
        
        user_subvert = keras.utils.to_categorical(user_subvert,len(subcategory_dict)+1)
        user_subvert = user_subvert.transpose((0,2,1))
        user_subvert = user_subvert[:,1:,:]
        user_subvert_mask = user_subvert.sum(axis=-1)
        
        user_vert_num = np.array(user_vert.sum(axis=-1),dtype='int32')
        user_subvert_num = np.array(user_subvert.sum(axis=-1),dtype='int32')
        
        user_subvert_mask = np.array(user_subvert_mask>0,dtype='float32')
        user_vert_mask = np.array(user_vert_mask>0,dtype='float32')
        vert_subvert_mask_input = np.array(vert_subvert_mask_input>0,dtype='float32')

        return [user_title, user_vert,user_vert_mask,user_subvert,user_subvert_mask,vert_subvert_mask_input,user_vert_num,user_subvert_num]

# %%


# %%


# %%
def evaluate_combine2(test_impressions,users,user_subvert_rep,user_vert_rep,user_global_rep,w1,w2,w3):
    AUC = []
    MRR = []
    nDCG5 = []
    nDCG10 =[]
    for i in range(len(test_impressions)):
        labels = test_impressions[i]['labels']
        nids = test_impressions[i]['docs']
        verts = news_vert[nids]
        verts = verts-1
        subverts = news_subvert[nids]
        subverts = subverts-1

        user_gv = user_global_rep[i]
        user_vv = user_vert_rep[i]
        user_svv = user_subvert_rep[i]

        click = users[i]
        
        nv = news_scoring[nids]
        score1 = np.dot(nv,user_gv)
        user_vv = user_vv[verts]
        score2 = (nv*user_vv).sum(axis=-1)
        
        mask2 = []
        for v in verts:
            t = news_vert[click]==(v+1)
            mask2.append(t.sum())
        mask2 = np.array(mask2)
        mask2 = mask2/((click>0).sum()+10**(-6))
        
        
        user_svv = user_svv[subverts]
        score3 = (nv*user_svv).sum(axis=-1)

        mask3 = []
        for svi in range(len(subverts)):
            sv = subverts[svi]
            t = (news_subvert[click]==(sv+1))
            mask3.append(t.sum())
        mask3 = np.array(mask3)
        mask3 = mask3/((click>0).sum()+10**(-6))
        

            
        score1 = np.array(score1)
        score2 = np.array(score2)
        score3 = np.array(score3)

        score = score1*w1+mask2*score2*w2+mask3*score3*w3
        

        auc = roc_auc_score(labels,score)
        mrr = mrr_score(labels,score)
        ndcg5 = ndcg_score(labels,score,k=5)
        ndcg10 = ndcg_score(labels,score,k=10)
    
        AUC.append(auc)
        MRR.append(mrr)
        nDCG5.append(ndcg5)
        nDCG10.append(ndcg10)

    return AUC, MRR, nDCG5, nDCG10

# %%


# %%
Res = []
# print(train_user['click'][0:10])
# exit(0)
# %%
model,news_encoder,user_encoder,rews = create_model(category_dict,subcategory_dict,title_word_embedding_matrix,entity_emb_matrix)
Res.append({'AUC':[],'MRR':[],'nDCG5':[],'nDCG10':[]})
train_generator = get_hir_train_generator(0.9999,news_info,index2nid,news_vert,news_subvert,news_entity_np,news_entity,train_user['click'],train_user_id,train_sess,train_label,16)
model.fit_generator(train_generator,epochs=4,verbose=3)

for i in range(1):
    model.fit_generator(train_generator,epochs=1,verbose=2)
    news_scoring = news_encoder.predict(news_info,verbose=True)
    test_user_generator = get_hir_user_generator(news_scoring,news_vert,news_subvert,news_entity_np,test_user['click'],32)
    
    AUC = []
    MRR = []
    nDCG5 = []
    nDCG10 = []

    for i in range(int(np.ceil(len(test_user['click'])/1000))):
        start = i*1000
        ed = (i+1)*1000
        ed = min(ed,len(test_user['click']))
        test_user_generator = get_hir_user_generator(news_scoring,news_vert,news_subvert,news_entity_np,test_user['click'][start:ed],32)
        user_subvert_rep,user_vert_rep,user_global_rep = user_encoder.predict_generator(test_user_generator,verbose=False)
        a,m,n5,n10 = evaluate_combine2(test_impressions[start:ed],test_user['click'][start:ed],user_subvert_rep,user_vert_rep,user_global_rep,0.15,0.15,0.7)
        AUC += a
        MRR += m
        nDCG5 += n5
        nDCG10 += n10

        print(np.array(AUC).mean(),np.array(MRR).mean(),np.array(nDCG5).mean(),np.array(nDCG10).mean())



    #     break
    AUC = np.array(AUC)
    MRR = np.array(MRR)
    nDCG5 = np.array(nDCG5)
    nDCG10 = np.array(nDCG10)

    AUC = AUC.mean()
    MRR = MRR.mean()
    nDCG5 = nDCG5.mean()
    nDCG10 = nDCG10.mean()
    
    Res[-1]['AUC'].append(AUC)
    Res[-1]['MRR'].append(MRR)
    Res[-1]['nDCG5'].append(nDCG5)
    Res[-1]['nDCG10'].append(nDCG10)

# %%
Res

# %%


# %%


# %%


# %%


# %%


# %%


# %%



