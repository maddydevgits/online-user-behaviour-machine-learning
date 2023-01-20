import pickle
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

dataset=pd.read_csv('online_classroom_data.csv')

model=open('model.pkl','rb')
classifier=pickle.load(model)

st.title('CLASSIFY ONLINE USER BEHAVIOR USING MACHINE LEARNING')

total_posts=st.number_input('Enter total number of total_posts:')
helpful_post=st.number_input('Enter total number of helpful_posts:')
nice_code_post=st.number_input('Enter total number of nice_code_posts:')
collaborative_post=st.number_input('Enter total number of collaborative_posts:')
confused_post=st.number_input('Enter total number of confused_posts:')
creative_post=st.number_input('Enter total number of creative_posts:')
bad_post=st.number_input('Enter total number of bad_posts:')
amazing_post=st.number_input('Enter total number ofamazing_ posts:')
timeonline=st.number_input('Enter total onlinetime:')

if st.button('classify'):
    data=[[total_posts,helpful_post,nice_code_post,collaborative_post,confused_post,creative_post,bad_post,amazing_post,timeonline]]
    x=dataset.iloc[:,1:10].values
    row_n=x.shape[0]
    x=np.insert(x,row_n,data,axis=0)
    ss=StandardScaler()
    x=ss.fit_transform(x)
    data=[x[-1]]
    result=classifier.predict(data)
    if result[0]==1:
        st.success('SUCCESS')
    else:
        st.error('FAILURE')
    


