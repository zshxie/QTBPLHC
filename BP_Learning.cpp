
#include "widget.h"
#include "ui_widget.h"










// NeuralNet.cpp: implementation of theCNeuralNet class.
//
//////////////////////////////////////////////////////////////////////
#include "stdafx.h"
#include "NeuralNet.h"
#include <assert.h>

CNeuralNet::CNeuralNet(int nInput, intnOutput, int nNeuronsPerLyr, int nHiddenLayer){
  assert(nInput>0 && nOutput>0 && nNeuronsPerLyr>0 &&nHiddenLayer>0 );
  m_nInput= nInput;
  m_nOutput= nOutput;
  m_nNeuronsPerLyr= nNeuronsPerLyr;
  if(nHiddenLayer!= 1)
     m_nHiddenLayer= 1;
  else
     m_nHiddenLayer= nHiddenLayer; //temporarily surpport only one hidden layer

  m_pHiddenLyr= NULL;
  m_pOutLyr= NULL;

  CreateNetwork();   //allocate for each layer
  InitializeNetwork();  //initialize the whole network
}

CNeuralNet::~CNeuralNet(){
  if(m_pHiddenLyr!= NULL)
     deletem_pHiddenLyr;
  if(m_pOutLyr!= NULL)
     deletem_pOutLyr;
}

void CNeuralNet::CreateNetwork(){
  m_pHiddenLyr= new SNeuronLayer(m_nNeuronsPerLyr, m_nInput);
  m_pOutLyr= new SNeuronLayer(m_nOutput, m_nNeuronsPerLyr);
}

void CNeuralNet::InitializeNetwork(){
  int i, j;  //variables for loop

  /*usepresent time as random seed, so every time runs this programm can producedifferent random sequence*/
  //srand((unsigned)time(NULL) );

  /*initializehidden layer's weights*/
  for(i=0;i<m_pHiddenLyr->m_nNeuron; i++){
     for(j=0;j<m_pHiddenLyr->m_pNeurons[i].m_nInput; j++){
       m_pHiddenLyr->m_pNeurons[i].m_pWeights[j]= RandomClamped();
       #ifdef NEED_MOMENTUM
       /*when the first epoch train started, there is no previous weights update*/
       m_pHiddenLyr->m_pNeurons[i].m_pPrevUpdate[j]= 0;
       #endif
     }
  }
  /*initialize output layer's weights*/
  for(i=0;i<m_pOutLyr->m_nNeuron; i++){
     for(int j=0; j<m_pOutLyr->m_pNeurons[i].m_nInput; j++){
       m_pOutLyr->m_pNeurons[i].m_pWeights[j]= RandomClamped();
       #ifdef NEED_MOMENTUM
       /*whenthe first epoch train started, there is no previous weights update*/
       m_pOutLyr->m_pNeurons[i].m_pPrevUpdate[j]= 0;
       #endif
     }
  }

  m_dErrorSum= 9999.0;  //initialize a large trainingerror, it will be decreasing with training
}

boolCNeuralNet::CalculateOutput(vector<double> input,vector<double>& output){
  if(input.size()!= m_nInput){ //input feature vector's dimention not equals to input of network
     return false;
  }
  inti, j;
  double nInputSum;  //sum term

  /*compute hidden layer output*/
  for(i=0;i<m_pHiddenLyr->m_nNeuron; i++){
     nInputSum= 0;
     for(j=0;j<m_pHiddenLyr->m_pNeurons[i].m_nInput-1; j++){
       nInputSum += m_pHiddenLyr->m_pNeurons[i].m_pWeights[j] * input[j];
     }
     /*plus bias term*/
     nInputSum += m_pHiddenLyr->m_pNeurons[i].m_pWeights[j] * BIAS;
     /*computesigmoid fuction's output*/
     m_pHiddenLyr->m_pNeurons[i].m_dActivation= Sigmoid(nInputSum);
  }

  /*compute output layer's output*/
  for(i=0;i<m_pOutLyr->m_nNeuron; i++){
     nInputSum= 0;
     for(j=0;j<m_pOutLyr->m_pNeurons[i].m_nInput-1; j++){
       nInputSum+= m_pOutLyr->m_pNeurons[i].m_pWeights[j]
         *m_pHiddenLyr->m_pNeurons[j].m_dActivation;
     }
     /*plusbias term*/
     nInputSum+= m_pOutLyr->m_pNeurons[i].m_pWeights[j] * BIAS;
     /*computesigmoid fuction's output*/
     m_pOutLyr->m_pNeurons[i].m_dActivation= Sigmoid(nInputSum);
     /*saveit to the output vector*/
     output.push_back(m_pOutLyr->m_pNeurons[i].m_dActivation);
  }
  returntrue;
}

bool CNeuralNet::TrainingEpoch(vector<iovector>&SetIn, vector<iovector>& SetOut, double LearningRate){
  inti, j, k;
  double WeightUpdate;  //weight's update value
  double err;  //error term

  /*increment'sgradient decrease(update weights according to each training sample)*/
  m_dErrorSum= 0;  // sum of error term
  for(i=0;i<SetIn.size(); i++){
     iovector vecOutputs;
     /*forwardlyspread inputs through network*/
     if(!CalculateOutput(SetIn[i],vecOutputs)){
       return false;
     }

     /*updatethe output layer's weights*/
     for(j=0;j<m_pOutLyr->m_nNeuron; j++){

       /*computeerror term*/
       err= ((double)SetOut[i][j]-vecOutputs[j])*vecOutputs[j]*(1-vecOutputs[j]);  //??隐层与输出层的局部梯度 2016 6 1 zshxie
       m_pOutLyr->m_pNeurons[j].m_dError= err;  //record this unit's error

       /*updatesum error*/
       m_dErrorSum+= ((double)SetOut[i][j] - vecOutputs[j]) * ((double)SetOut[i][j] -vecOutputs[j]);//(dk-yk)^2

       /*updateeach input's weight*/
       for(k=0;k<m_pOutLyr->m_pNeurons[j].m_nInput-1; k++){
         WeightUpdate= err * LearningRate * m_pHiddenLyr->m_pNeurons[k].m_dActivation;//权重调整率 梯度*学习率*隐层净输入
#ifdef NEED_MOMENTUM   //需要动量项
         /*updateweights with momentum*/
         m_pOutLyr->m_pNeurons[j].m_pWeights[k]+=
            WeightUpdate+ m_pOutLyr->m_pNeurons[j].m_pPrevUpdate[k] * MOMENTUM;
         m_pOutLyr->m_pNeurons[j].m_pPrevUpdate[k]= WeightUpdate;
#else
         /*updateunit weights*/
         m_pOutLyr->m_pNeurons[j].m_pWeights[k]+= WeightUpdate;
#endif
       }
       /*biasupdate volume*/
       WeightUpdate= err * LearningRate * BIAS;
#ifdef NEED_MOMENTUM
       /*updatebias with momentum*/
       m_pOutLyr->m_pNeurons[j].m_pWeights[k]+=
         WeightUpdate+ m_pOutLyr->m_pNeurons[j].m_pPrevUpdate[k] * MOMENTUM;
       m_pOutLyr->m_pNeurons[j].m_pPrevUpdate[k]= WeightUpdate;
#else
       /*updatebias*/
       m_pOutLyr->m_pNeurons[j].m_pWeights[k]+= WeightUpdate;
#endif
     }//for out layer

     /*updatethe hidden layer's weights*/
     for(j=0;j<m_pHiddenLyr->m_nNeuron; j++){

       err= 0;
       for(intk=0; k<m_pOutLyr->m_nNeuron; k++){
         err+= m_pOutLyr->m_pNeurons[k].m_dError *m_pOutLyr->m_pNeurons[k].m_pWeights[j];
       }
       err*= m_pHiddenLyr->m_pNeurons[j].m_dActivation * (1 -m_pHiddenLyr->m_pNeurons[j].m_dActivation);
       m_pHiddenLyr->m_pNeurons[j].m_dError= err;  //record this unit's error

       /*updateeach input's weight*/
       for(k=0;k<m_pHiddenLyr->m_pNeurons[j].m_nInput-1; k++){
         WeightUpdate= err * LearningRate * SetIn[i][k];
#ifdef NEED_MOMENTUM
         /*updateweights with momentum*/
         m_pHiddenLyr->m_pNeurons[j].m_pWeights[k]+=
            WeightUpdate+ m_pHiddenLyr->m_pNeurons[j].m_pPrevUpdate[k] * MOMENTUM;
         m_pHiddenLyr->m_pNeurons[j].m_pPrevUpdate[k]= WeightUpdate;
#else
         m_pHiddenLyr->m_pNeurons[j].m_pWeights[k]+= WeightUpdate;
#endif
       }
       /*biasupdate volume*/
       WeightUpdate= err * LearningRate * BIAS;
#ifdef NEED_MOMENTUM
       /*updatebias with momentum*/
       m_pHiddenLyr->m_pNeurons[j].m_pWeights[k]+=
         WeightUpdate+ m_pHiddenLyr->m_pNeurons[j].m_pPrevUpdate[k] * MOMENTUM;
       m_pHiddenLyr->m_pNeurons[j].m_pPrevUpdate[k]= WeightUpdate;
#else
       /*updatebias*/
       m_pHiddenLyr->m_pNeurons[j].m_pWeights[k]+= WeightUpdate;
#endif
     }//forhidden layer
  }//forone epoch
  returntrue;
}



boolTrain(vector<iovector>& SetIn, vector<iovector>& SetOut);

bool SaveTrainResultToFile(const char* lpszFileName, boolbCreate);

bool LoadTrainResultFromFile(const char* lpszFileName, DWORDdwStartPos);

int Recognize(CString strPathName, CRect rt, double&dConfidence);



头文件：
#ifndef __OPERATEONNEURALNET_H__
#define __OPERATEONNEURALNET_H__

#include "NeuralNet.h"
#define NEURALNET_VERSION 1.0
#define RESAMPLE_LEN 4

class COperateOnNeuralNet{
private:
  /*network*/
  CNeuralNet*m_oNetWork;

  /*network'sparameter*/
  int m_nInput;
  int m_nOutput;
  int m_nNeuronsPerLyr;
  int m_nHiddenLayer;

  /*trainingconfiguration*/
  int m_nMaxEpoch;  // max training epoch times
  double m_dMinError; // error threshold
  double m_dLearningRate;

  /*dinamiccurrent parameter*/
  int m_nEpochs;
  double m_dErr;  //mean error of oneepoch(m_dErrorSum/(num-of-samples * num-of-output))
  bool m_bStop; //control whether stop or not during the training
  vector<double> m_vecError;   //record each epoch'straining error, used for drawing error curve

public:
  COperateOnNeuralNet();
  ~COperateOnNeuralNet();

  voidSetNetWorkParameter(int nInput, int nOutput, int nNeuronsPerLyr, intnHiddenLayer);
  boolCreatNetWork();
  voidSetTrainConfiguration(int nMaxEpoch, double dMinError, double dLearningRate);
  voidSetStopFlag(bool bStop) { m_bStop = bStop; }

  doubleGetError(){ return m_dErr; }
  intGetEpoch(){ return m_nEpochs; }
  intGetNumNeuronsPerLyr(){ return m_nNeuronsPerLyr; }

  boolTrain(vector<iovector>& SetIn, vector<iovector>& SetOut);
  bool SaveTrainResultToFile(const char* lpszFileName, boolbCreate);
  bool LoadTrainResultFromFile(const char* lpszFileName, DWORDdwStartPos);
  int Recognize(CString strPathName, CRect rt, double&dConfidence);
};
/*
* Can be used when saving or readingtraining result.
*/
struct NEURALNET_HEADER{
  DWORD dwVersion;  //version imformation

  /*initialparameters*/
  int m_nInput;  //number of inputs
  int m_nOutput; //number of outputs
  int m_nNeuronsPerLyr; //unit number of hidden layer
  int m_nHiddenLayer; //hidden layer, not including the output layer

  /*trainingconfiguration*/
  int m_nMaxEpoch;  // max training epoch times
  double m_dMinError; // error threshold
  double m_dLearningRate;

  /*dinamiccurrent parameter*/
  int m_nEpochs;
  double m_dErr;  //mean error of oneepoch(m_dErrorSum/(num-of-samples * num-of-output))
};
#endif //__OPERATEONNEURALNET_H__


// OperateOnNeuralNet.cpp: implementationof the COperateOnNeuralNet class.
//
//////////////////////////////////////////////////////////////////////

#include "stdafx.h"
#include "MyDigitRec.h"
#include "OperateOnNeuralNet.h"
#include "Img.h"
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

voidCOperateOnNeuralNet::SetNetWorkParameter(int nInput, int nOutput, intnNeuronsPerLyr, int nHiddenLayer){
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
     returntrue;
  else
     returnfalse;
}

voidCOperateOnNeuralNet::SetTrainConfiguration(int nMaxEpoch, double dMinError,double dLearningRate){
  assert(nMaxEpoch>0&& !(dMinError<0) && dLearningRate!=0);
  m_nMaxEpoch= nMaxEpoch;
  m_dMinError= dMinError;
  m_dLearningRate= dLearningRate;
}

boolCOperateOnNeuralNet::Train(vector<iovector>& SetIn, vector<iovector>&SetOut){
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

boolCOperateOnNeuralNet::SaveTrainResultToFile(const char* lpszFileName, boolbCreate){
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
  NEURALNET_HEADERheader = {0};
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

boolCOperateOnNeuralNet::LoadTrainResultFromFile(const char* lpszFileName, DWORDdwStartPos){
  CFilefile;
  if(!file.Open(lpszFileName,CFile::modeRead)){
     returnfalse;
  }

  file.Seek(dwStartPos,CFile::begin);  //point to dwStartPos

  /*readin NeuralNet_Head infomation*/
  NEURALNET_HEADERheader = {0};
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
  m_dErr= header.m_dErr;		  //update training error

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

