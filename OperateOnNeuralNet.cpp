#include "widget.h"
#include "ui_widget.h"
// OperateOnNeuralNet.cpp: implementationof the COperateOnNeuralNet class.

//

//////////////////////////////////////////////////////////////////////



//#include "stdafx.h"

//#include "MyDigitRec.h"

#include "OperateOnNeuralNet.h"

//#include "Img.h"

#include <assert.h>



/*

*Handle message during waiting.

*/

void WaitForIdle()

{

         MSGmsg;

         while(::PeekMessage(&msg,NULL, 0, 0, PM_REMOVE))

         {

                   ::TranslateMessage(&msg);

                   ::DispatchMessage(&msg);

         }

}



COperateOnNeuralNet::COperateOnNeuralNet(){

         m_nInput= 0;

         m_nOutput= 0;

         m_nNeuronsPerLyr= 0;

         m_nHiddenLayer= 0;



         m_nMaxEpoch= 0;

         m_dMinError= 0;

         m_dLearningRate= 0;



         m_oNetWork= 0;



         m_nEpochs= 0;

         m_dErr= 0;

         m_bStop= false;

}



COperateOnNeuralNet::~COperateOnNeuralNet(){

         if(m_oNetWork)

                   deletem_oNetWork;

}



void COperateOnNeuralNet::SetNetWorkParameter(int nInput, int nOutput, intnNeuronsPerLyr, int nHiddenLayer){

         assert(nInput>0 && nOutput>0 && nNeuronsPerLyr>0 &&nHiddenLayer>0 );

         m_nInput= nInput;

         m_nOutput= nOutput;

         m_nNeuronsPerLyr= nNeuronsPerLyr;

         m_nHiddenLayer= nHiddenLayer;

}



bool COperateOnNeuralNet::CreatNetWork(){

         assert(m_nInput>0 && m_nOutput>0 && m_nNeuronsPerLyr>0&& m_nHiddenLayer>0 );

         m_oNetWork= new CNeuralNet(m_nInput, m_nOutput, m_nNeuronsPerLyr, m_nHiddenLayer);

         if(m_oNetWork)

                   return true;

         else

                   return false;

}



void COperateOnNeuralNet::SetTrainConfiguration(int nMaxEpoch, double dMinError,double dLearningRate){

         assert(nMaxEpoch>0&& !(dMinError<0) && dLearningRate!=0);

         m_nMaxEpoch= nMaxEpoch;

         m_dMinError= dMinError;

         m_dLearningRate= dLearningRate;

}



bool COperateOnNeuralNet::Train(vector<iovector>& SetIn, vector<iovector>&SetOut){

         m_bStop= false;  //no stop during training

         CStringstrOutMsg;



         do{

                   /*trainone epoch*/

                   if(!m_oNetWork->TrainingEpoch(SetIn, SetOut, m_dLearningRate) ){

                            strOutMsg.Format("Erroroccured at training %dth epoch!",m_nEpochs+1);

                            AfxMessageBox(strOutMsg);

                            returnfalse;

                   }else{

                            m_nEpochs++;

                   }



                   /*computemean error of one epoch(m_dErrorSum/(num-of-samples * num-of-output))*/

                   intsum = m_oNetWork->GetErrorSum();

                   m_dErr= m_oNetWork->GetErrorSum() / ( m_nOutput * SetIn.size() );

                   m_vecError.push_back(m_dErr);

                   if(m_dErr< m_dMinError){

                            break;

                   }



                   /*stopin loop to chech wether user's action made or message sent, mostly for changem_bStop */

                   WaitForIdle();



                   if(m_bStop){

                            break;

                   }

         }while(--m_nMaxEpoch> 0);



         returntrue;

}



bool COperateOnNeuralNet::SaveTrainResultToFile(const char* lpszFileName, boolbCreate){

         CFilefile;

         if(bCreate){

                   if(!file.Open(lpszFileName,CFile::modeWrite|CFile::modeCreate))

                            returnfalse;

         }else{

                   if(!file.Open(lpszFileName,CFile::modeWrite))

                            returnfalse;

                   file.SeekToEnd();  //add to end of file

         }



         /*createnetwork head information*/

         /*initialparameter*/

         NEURALNET_HEADER header = {0};

         header.dwVersion= NEURALNET_VERSION;

         header.m_nInput= m_nInput;

         header.m_nOutput= m_nOutput;

         header.m_nNeuronsPerLyr= m_nNeuronsPerLyr;

         header.m_nHiddenLayer= m_nHiddenLayer;

         /*trainingconfiguration*/

         header.m_nMaxEpoch= m_nMaxEpoch;

         header.m_dMinError= m_dMinError;

         header.m_dLearningRate= m_dLearningRate;

         /*dinamiccurrent parameter*/

         header.m_nEpochs= m_nEpochs;

         header.m_dErr= m_dErr;



         file.Write(&header,sizeof(header));



         /*writeweight information to file*/

         inti, j;

         /*hiddenlayer weight*/

         for(i=0;i<m_oNetWork->GetHiddenLyr()->m_nNeuron; i++){

                   file.Write(&m_oNetWork->GetHiddenLyr()->m_pNeurons[i].m_dActivation,

                            sizeof(m_oNetWork->GetHiddenLyr()->m_pNeurons[i].m_dActivation));

                   file.Write(&m_oNetWork->GetHiddenLyr()->m_pNeurons[i].m_dError,

                            sizeof(m_oNetWork->GetHiddenLyr()->m_pNeurons[i].m_dError));

                   for(j=0;j<m_oNetWork->GetHiddenLyr()->m_pNeurons[i].m_nInput; j++){

                            file.Write(&m_oNetWork->GetHiddenLyr()->m_pNeurons[i].m_pWeights[j],

                                     sizeof(m_oNetWork->GetHiddenLyr()->m_pNeurons[i].m_pWeights[j]));

                   }

         }

         /*outputlayer weight*/

         for(i=0;i<m_oNetWork->GetOutLyr()->m_nNeuron; i++){

                   file.Write(&m_oNetWork->GetOutLyr()->m_pNeurons[i].m_dActivation,

                            sizeof(m_oNetWork->GetOutLyr()->m_pNeurons[i].m_dActivation));

                   file.Write(&m_oNetWork->GetOutLyr()->m_pNeurons[i].m_dError,

                            sizeof(m_oNetWork->GetOutLyr()->m_pNeurons[i].m_dError));

                   for(j=0;j<m_oNetWork->GetOutLyr()->m_pNeurons[i].m_nInput; j++){

                            file.Write(&m_oNetWork->GetOutLyr()->m_pNeurons[i].m_pWeights[j],

                                     sizeof(m_oNetWork->GetOutLyr()->m_pNeurons[i].m_pWeights[j]));

                   }

         }



         file.Close();

         returntrue;

}



bool COperateOnNeuralNet::LoadTrainResultFromFile(const char* lpszFileName, DWORDdwStartPos){

         CFilefile;

         if(!file.Open(lpszFileName,CFile::modeRead)){

                   returnfalse;

         }



         file.Seek(dwStartPos,CFile::begin);  //point to dwStartPos



         /*readin NeuralNet_Head infomation*/

         NEURALNET_HEADER header = {0};

         if(file.Read(&header, sizeof(header)) != sizeof(header) ){

                   returnfalse;

         }



         /*chechversion*/

         if(header.dwVersion!= NEURALNET_VERSION){

                   returnfalse;

         }



         /*checkbasic NeuralNet's structure*/

         if(header.m_nInput!= m_nInput

                   ||header.m_nOutput != m_nOutput

                   ||header.m_nNeuronsPerLyr != m_nNeuronsPerLyr

                   ||header.m_nHiddenLayer != m_nHiddenLayer

                   ||header.m_nMaxEpoch != m_nMaxEpoch

                   ||header.m_dMinError != m_dMinError

                   ||header.m_dLearningRate != m_dLearningRate ){

                   returnfalse;

         }



         /*dynamicparameters*/

         m_nEpochs= header.m_nEpochs;  //update trainingepochs

         m_dErr= header.m_dErr;                    //update training error



         /*readin NetWork's weights*/

         inti,j;

         /*readin hidden layer weights*/

         for(i=0;i<m_oNetWork->GetHiddenLyr()->m_nNeuron; i++){

                   file.Read(&m_oNetWork->GetHiddenLyr()->m_pNeurons[i].m_dActivation,

                            sizeof(m_oNetWork->GetHiddenLyr()->m_pNeurons[i].m_dActivation));

                   file.Read(&m_oNetWork->GetHiddenLyr()->m_pNeurons[i].m_dError,

                            sizeof(m_oNetWork->GetHiddenLyr()->m_pNeurons[i].m_dError));



                   for(j=0;j<m_oNetWork->GetHiddenLyr()->m_pNeurons[i].m_nInput; j++){

                            file.Read(&m_oNetWork->GetHiddenLyr()->m_pNeurons[i].m_pWeights[j],

                                     sizeof(m_oNetWork->GetHiddenLyr()->m_pNeurons[i].m_pWeights[j]));

                   }

         }



         /*readin out layer weights*/

         for(i=0;i<m_oNetWork->GetOutLyr()->m_nNeuron; i++){

                   file.Read(&m_oNetWork->GetOutLyr()->m_pNeurons[i].m_dActivation,

                            sizeof(m_oNetWork->GetOutLyr()->m_pNeurons[i].m_dActivation));

                   file.Read(&m_oNetWork->GetOutLyr()->m_pNeurons[i].m_dError,

                            sizeof(m_oNetWork->GetOutLyr()->m_pNeurons[i].m_dError));



                   for(j=0;j<m_oNetWork->GetOutLyr()->m_pNeurons[i].m_nInput; j++){

                            file.Read(&m_oNetWork->GetOutLyr()->m_pNeurons[i].m_pWeights[j],

                                     sizeof(m_oNetWork->GetOutLyr()->m_pNeurons[i].m_pWeights[j]));

                   }

         }



         returntrue;

}



int COperateOnNeuralNet::Recognize(CStringstrPathName, CRect rt, double &dConfidence){

         intnBestMatch;  //category number

         doubledMaxOut1 = 0; //max output

         doubledMaxOut2 = 0; //second max output



         CImggray;

         if(!gray.AttachFromFile(strPathName)){

                   return-1;

         }



         /*convert the picture waitiong for being recognized to vector*/

         vector<double>vecToRec;

         for(intj=rt.top; j<rt.bottom; j+= RESAMPLE_LEN){

                   for(inti=rt.left; i<rt.right; i+=RESAMPLE_LEN){

                            intnGray = 0;

                            for(intmm=j; mm<j+RESAMPLE_LEN; mm++){

                                     for(intnn=i; nn<i+RESAMPLE_LEN; nn++)

                                               nGray+= gray.GetGray(nn, mm);

                            }

                            nGray/= RESAMPLE_LEN*RESAMPLE_LEN;

                            vecToRec.push_back(nGray/255.0);

                   }

         }



         /*computethe output result*/

         vector<double>outputs;

         if(!m_oNetWork->CalculateOutput(vecToRec,outputs)){

                   AfxMessageBox("Recfailed!");

                   return-1;

         }



         /*findthe max output unit, and its unit number is the category number*/

         nBestMatch= 0;

         for(intk=0; k<outputs.size(); k++){

                   if(outputs[k]> dMaxOut1){

                            dMaxOut2= dMaxOut1;

                            dMaxOut1= outputs[k];

                            nBestMatch= k;

                   }

         }

         dConfidence= dMaxOut1 - dMaxOut2;  //compute beliefdegree

         returnnBestMatch;

}
