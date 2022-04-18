import streamlit as st
import pandas as pd
import numpy as np
#import sklearn
#from sklearn.ensemble import RandomForestClassifier
import pickle
import base64

#streamlit
def main():        
    
    st.set_page_config(page_title = 'Propensity to Purchase',\
                       #page_icon = 'logo_dh.jpeg',
                       layout='wide',
                       initial_sidebar_state = 'expanded')
    
    c1, c2 = st.columns([3,1])
    c1.title('Propensity to purchase')
    #c2.write('Teste Commit')
    with st.expander('App description',expanded=True):
        st.markdown('The main objective of this tool is to make predictions about the chance of a customer converting in a given marketing campaign...')
    
#################################################################################################################
    feature_importance_top15 = ['last_purchase_days',
                                'part_history',
                                'hist_customer_8',
                                'prod_type_b',
                                'hist_customer_5',
                                'prod_type_e',
                                'income',
                                'years_as_customer',
                                'hist_customer_3',
                                'prod_type_a',
                                'hist_customer_4',
                                'children',
                                'hist_customer_1',
                                'hist_customer_2',
                                'age']

    #mdl_rfc_best = pickle.load(open('pickle_mdl_rfc_best.sav', 'rb'))
    Xtrain_selec = pickle.load(open('pickle_Xtrain_selec.sav', 'rb'))
    ytrain = pickle.load(open('pickle_ytrain.sav', 'rb'))
    #st.dataframe(Xtrain_selec)
    
#################################################################################################################
    with st.sidebar:
        st.title('Canada coupons')
        database = st.radio('Input data (X):',('Manual', 'CSV'))
        
        if database == 'CSV':
            st.info('Upload do CSV')
            file = st.file_uploader('Select the CSV file containing the columns described above',type='csv')
            if file:
                Xtest_selec = pd.read_csv(file)
        else:
            X0 = st.slider(feature_importance_top15[0],0,100,step=1)
            X1 = st.slider(feature_importance_top15[1],0,4,step=1)
            X2 = st.slider(feature_importance_top15[2],0,1,step=1)
            X3 = st.slider(feature_importance_top15[3],0,1500,step=100)
            X4 = st.slider(feature_importance_top15[4],0,35,step=1)
            X5 = st.slider(feature_importance_top15[5],0,600,step=50)
            X6 = st.slider(feature_importance_top15[6],0,200000,step=1000)
            X7 = st.slider(feature_importance_top15[7],1,6,step=1)
            X8 = st.slider(feature_importance_top15[8],0,40,step=1)
            X9 = st.slider(feature_importance_top15[9],0,2500,step=50)
            X10 = st.slider(feature_importance_top15[10],0,20,step=1)
            X11 = st.slider(feature_importance_top15[11],0,1,step=1)
            X12 = st.slider(feature_importance_top15[12],0,35,step=1)
            X13 = st.slider(feature_importance_top15[13],0,35,step=1)
            X14 = st.slider(feature_importance_top15[14],18,90,step=1)

            Xtest_selec = pd.DataFrame({    'last_purchase_days':[X0],
                                            'part_history':[X1],
                                            'hist_customer_8':[X2],
                                            'prod_type_b':[X3],
                                            'hist_customer_5':[X4],
                                            'prod_type_e':[X5],
                                            'income':[X6],
                                            'years_as_customer':[X7],
                                            'hist_customer_3':[X8],
                                            'prod_type_a':[X9],
                                            'hist_customer_4':[X10],
                                            'children':[X11],
                                            'hist_customer_1':[X12],
                                            'hist_customer_2':[X13],
                                            'age':[X14]})
                                     
##################################################################################################################

    mdl_rfc_best = RandomForestClassifier(
                             n_estimators= 500, 
                             min_samples_split= 5, 
                             min_samples_leaf= 2, 
                             max_features= 'auto',
                             max_depth= 15, 
                             class_weight= 'balanced', 
                             bootstrap= True)

    mdl_rfc_best.fit(Xtrain_selec, ytrain)

    ypred_rfc = mdl_rfc_best.predict(Xtest_selec.to_numpy())


##################################################################################################################    

    if database == 'Manual':
        with st.expander('View Input Data', expanded = False):
                st.dataframe(Xtest_selec.T)
        with st.expander('View Predictions', expanded = False):
                if ypred_rfc == 0:
                    st.error(ypred_rfc[0])
                else:
                    st.success(ypred_rfc[0])
                    
        if st.button('Download csv'):
            df_download = Xtest_selec.copy()
            df_download['Response_pred'] = ypred_rfc
            st.dataframe(df_download)
            csv = df_download.to_csv(sep=',',decimal=',',index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a>'
            st.markdown(href, unsafe_allow_html=True)
            

    else: #database == 'CSV'
        if file:
            with st.expander('View Input Data', expanded = False):
                st.dataframe(Xtest_selec)
            with st.expander('View Predictions', expanded = False):
                st.dataframe(ypred_rfc)            
            
            if st.button('Download csv'):
                df_download = Xtest_selec.copy()
                df_download['Response_pred'] = ypred_rfc
                st.write(df_download.shape)
                st.dataframe(df_download)
                csv = df_download.to_csv(sep=',',decimal=',',index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a>'
                st.markdown(href, unsafe_allow_html=True)

if __name__ == '__main__':
	main()
