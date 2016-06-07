//����λ��������תΪһλʮ������
/*
03.#include<iostream>
04.#include<cmath>
05.using namespace std;
06.
07.#define  innode 3  //��������
08.#define  hidenode 10//���������
09.#define  outnode 1 //��������
10.#define  trainsample 8//BPѵ��������
11.class BpNet
12.{
13.public:
14.    void train(double p[trainsample][innode ],double t[trainsample][outnode]);//Bpѵ��
15.    double p[trainsample][innode];     //���������
16.    double t[trainsample][outnode];    //����Ҫ�����
17.
18.    double *recognize(double *p);//Bpʶ��
19.
20.    void writetrain(); //дѵ�����Ȩֵ
21.    void readtrain(); //��ѵ���õ�Ȩֵ����ʹ�Ĳ���ÿ��ȥѵ���ˣ�ֻҪ��ѵ����õ�Ȩֵ��������OK
22.
23.    BpNet();
24.    virtual ~BpNet();
25.
26.public:
27.    void init();
28.    double w[innode][hidenode];//�������Ȩֵ
29.    double w1[hidenode][outnode];//������Ȩֵ
30.    double b1[hidenode];//������㷧ֵ
31.    double b2[outnode];//�����㷧ֵ
32.
33.    double rate_w; //Ȩֵѧϰ�ʣ������-������)
34.    double rate_w1;//Ȩֵѧϰ�� (������-�����)
35.    double rate_b1;//�����㷧ֵѧϰ��
36.    double rate_b2;//����㷧ֵѧϰ��
37.
38.    double e;//������
39.    double error;//�����������
40.    double result[outnode];// Bp���
41.};
42.
43.BpNet::BpNet()
44.{
45.    error=1.0;
46.    e=0.0;
47.
48.    rate_w=0.9;  //Ȩֵѧϰ�ʣ������--������)
49.    rate_w1=0.9; //Ȩֵѧϰ�� (������--�����)
50.    rate_b1=0.9; //�����㷧ֵѧϰ��
51.    rate_b2=0.9; //����㷧ֵѧϰ��
52.}
53.
54.BpNet::~BpNet()
55.{
56.
57.}
58.
59.void winit(double w[],int n) //Ȩֵ��ʼ��
60.{
61.  for(int i=0;i<n;i++)
62.    w[i]=(2.0*(double)rand()/RAND_MAX)-1;
63.}
64.
65.void BpNet::init()
66.{
67.    winit((double*)w,innode*hidenode);
68.    winit((double*)w1,hidenode*outnode);
69.    winit(b1,hidenode);
70.    winit(b2,outnode);
71.}
72.
73.void BpNet::train(double p[trainsample][innode],double t[trainsample][outnode])
74.{
75.    double pp[hidenode];//��������У�����
76.    double qq[outnode];//ϣ�����ֵ��ʵ�����ֵ��ƫ��
77.    double yd[outnode];//ϣ�����ֵ
78.
79.    double x[innode]; //��������
80.    double x1[hidenode];//�������״ֵ̬
81.    double x2[outnode];//������״ֵ̬
82.    double o1[hidenode];//�����㼤��ֵ
83.    double o2[hidenode];//����㼤��ֵ
84.
85.    for(int isamp=0;isamp<trainsample;isamp++)//ѭ��ѵ��һ����Ʒ
86.    {
87.        for(int i=0;i<innode;i++)
88.            x[i]=p[isamp][i]; //���������
89.        for(int i=0;i<outnode;i++)
90.            yd[i]=t[isamp][i]; //�������������
91.
92.        //����ÿ����Ʒ������������׼
93.        for(int j=0;j<hidenode;j++)
94.        {
95.            o1[j]=0.0;
96.            for(int i=0;i<innode;i++)
97.                o1[j]=o1[j]+w[i][j]*x[i];//���������Ԫ���뼤��ֵ
98.            x1[j]=1.0/(1+exp(-o1[j]-b1[j]));//���������Ԫ�����
99.            //    if(o1[j]+b1[j]>0) x1[j]=1;
100.            //else x1[j]=0;
101.        }
102.
103.        for(int k=0;k<outnode;k++)
104.        {
105.            o2[k]=0.0;
106.            for(int j=0;j<hidenode;j++)
107.                o2[k]=o2[k]+w1[j][k]*x1[j]; //��������Ԫ���뼤��ֵ
108.            x2[k]=1.0/(1.0+exp(-o2[k]-b2[k])); //��������Ԫ���
109.            //    if(o2[k]+b2[k]>0) x2[k]=1;
110.            //    else x2[k]=0;
111.        }
112.
113.        for(int k=0;k<outnode;k++)
114.        {
115.            qq[k]=(yd[k]-x2[k])*x2[k]*(1-x2[k]); //ϣ�������ʵ�������ƫ��
116.            for(int j=0;j<hidenode;j++)
117.                w1[j][k]+=rate_w1*qq[k]*x1[j];  //��һ�ε�������������֮���������Ȩ
118.        }
119.
120.        for(int j=0;j<hidenode;j++)
121.        {
122.            pp[j]=0.0;
123.            for(int k=0;k<outnode;k++)
124.                pp[j]=pp[j]+qq[k]*w1[j][k];
125.            pp[j]=pp[j]*x1[j]*(1-x1[j]); //�������У�����
126.
127.            for(int i=0;i<innode;i++)
128.                w[i][j]+=rate_w*pp[j]*x[i]; //��һ�ε�������������֮���������Ȩ
129.        }
130.
131.        for(int k=0;k<outnode;k++)
132.        {
133.            e+=fabs(yd[k]-x2[k])*fabs(yd[k]-x2[k]); //���������
134.        }
135.        error=e/2.0;
136.
137.        for(int k=0;k<outnode;k++)
138.            b2[k]=b2[k]+rate_b2*qq[k]; //��һ�ε�������������֮�������ֵ
139.        for(int j=0;j<hidenode;j++)
140.            b1[j]=b1[j]+rate_b1*pp[j]; //��һ�ε�������������֮�������ֵ
141.    }
142.}
143.
144.double *BpNet::recognize(double *p)
145.{
146.    double x[innode]; //��������
147.    double x1[hidenode]; //�������״ֵ̬
148.    double x2[outnode]; //������״ֵ̬
149.    double o1[hidenode]; //�����㼤��ֵ
150.    double o2[hidenode]; //����㼤��ֵ
151.
152.    for(int i=0;i<innode;i++)
153.        x[i]=p[i];
154.
155.    for(int j=0;j<hidenode;j++)
156.    {
157.        o1[j]=0.0;
158.        for(int i=0;i<innode;i++)
159.            o1[j]=o1[j]+w[i][j]*x[i]; //���������Ԫ����ֵ
160.        x1[j]=1.0/(1.0+exp(-o1[j]-b1[j])); //���������Ԫ���
161.        //if(o1[j]+b1[j]>0) x1[j]=1;
162.        //    else x1[j]=0;
163.    }
164.
165.    for(int k=0;k<outnode;k++)
166.    {
167.        o2[k]=0.0;
168.        for(int j=0;j<hidenode;j++)
169.            o2[k]=o2[k]+w1[j][k]*x1[j];//��������Ԫ����ֵ
170.        x2[k]=1.0/(1.0+exp(-o2[k]-b2[k]));//��������Ԫ���
171.        //if(o2[k]+b2[k]>0) x2[k]=1;
172.        //else x2[k]=0;
173.    }
174.
175.    for(int k=0;k<outnode;k++)
176.    {
177.        result[k]=x2[k];
178.    }
179.    return result;
180.}
181.
182.void BpNet::writetrain()
183.{
184.    FILE *stream0;
185.    FILE *stream1;
186.    FILE *stream2;
187.    FILE *stream3;
188.    int i,j;
189.    //�������Ȩֵд��
190.    if(( stream0 = fopen("w.txt", "w+" ))==NULL)
191.    {
192.        cout<<"�����ļ�ʧ��!";
193.        exit(1);
194.    }
195.    for(i=0;i<innode;i++)
196.    {
197.        for(j=0;j<hidenode;j++)
198.        {
199.            fprintf(stream0, "%f\n", w[i][j]);
200.        }
201.    }
202.    fclose(stream0);
203.
204.    //������Ȩֵд��
205.    if(( stream1 = fopen("w1.txt", "w+" ))==NULL)
206.    {
207.        cout<<"�����ļ�ʧ��!";
208.        exit(1);
209.    }
210.    for(i=0;i<hidenode;i++)
211.    {
212.        for(j=0;j<outnode;j++)
213.        {
214.            fprintf(stream1, "%f\n",w1[i][j]);
215.        }
216.    }
217.    fclose(stream1);
218.
219.    //������㷧ֵд��
220.    if(( stream2 = fopen("b1.txt", "w+" ))==NULL)
221.    {
222.        cout<<"�����ļ�ʧ��!";
223.        exit(1);
224.    }
225.    for(i=0;i<hidenode;i++)
226.        fprintf(stream2, "%f\n",b1[i]);
227.    fclose(stream2);
228.
229.    //�����㷧ֵд��
230.    if(( stream3 = fopen("b2.txt", "w+" ))==NULL)
231.    {
232.        cout<<"�����ļ�ʧ��!";
233.        exit(1);
234.    }
235.    for(i=0;i<outnode;i++)
236.        fprintf(stream3, "%f\n",b2[i]);
237.    fclose(stream3);
238.
239.}
240.
241.void BpNet::readtrain()
242.{
243.    FILE *stream0;
244.    FILE *stream1;
245.    FILE *stream2;
246.    FILE *stream3;
247.    int i,j;
248.
249.    //�������Ȩֵ����
250.    if(( stream0 = fopen("w.txt", "r" ))==NULL)
251.    {
252.        cout<<"���ļ�ʧ��!";
253.        exit(1);
254.    }
255.    float  wx[innode][hidenode];
256.    for(i=0;i<innode;i++)
257.    {
258.        for(j=0;j<hidenode;j++)
259.        {
260.            fscanf(stream0, "%f", &wx[i][j]);
261.            w[i][j]=wx[i][j];
262.        }
263.    }
264.    fclose(stream0);
265.
266.    //������Ȩֵ����
267.    if(( stream1 = fopen("w1.txt", "r" ))==NULL)
268.    {
269.        cout<<"���ļ�ʧ��!";
270.        exit(1);
271.    }
272.    float  wx1[hidenode][outnode];
273.    for(i=0;i<hidenode;i++)
274.    {
275.        for(j=0;j<outnode;j++)
276.        {
277.            fscanf(stream1, "%f", &wx1[i][j]);
278.            w1[i][j]=wx1[i][j];
279.        }
280.    }
281.    fclose(stream1);
282.
283.    //������㷧ֵ����
284.    if(( stream2 = fopen("b1.txt", "r" ))==NULL)
285.    {
286.        cout<<"���ļ�ʧ��!";
287.        exit(1);
288.    }
289.    float xb1[hidenode];
290.    for(i=0;i<hidenode;i++)
291.    {
292.        fscanf(stream2, "%f",&xb1[i]);
293.        b1[i]=xb1[i];
294.    }
295.    fclose(stream2);
296.
297.    //�����㷧ֵ����
298.    if(( stream3 = fopen("b2.txt", "r" ))==NULL)
299.    {
300.        cout<<"���ļ�ʧ��!";
301.        exit(1);
302.    }
303.    float xb2[outnode];
304.    for(i=0;i<outnode;i++)
305.    {
306.        fscanf(stream3, "%f",&xb2[i]);
307.        b2[i]=xb2[i];
308.    }
309.    fclose(stream3);
310.}
311.
312.
313.//��������
314.double X[trainsample][innode]= {
315.    {0,0,0},{0,0,1},{0,1,0},{0,1,1},{1,0,0},{1,0,1},{1,1,0},{1,1,1}
316.    };
317.//�����������
318.double Y[trainsample][outnode]={
319.    {0},{0.1429},{0.2857},{0.4286},{0.5714},{0.7143},{0.8571},{1.0000}
320.    };
321.
322.int main()
323.{
324.    BpNet bp;
325.    bp.init();
326.    int times=0;
327.    while(bp.error>0.0001)
328.    {
329.        bp.e=0.0;
330.        times++;
331.        bp.train(X,Y);
332.        cout<<"Times="<<times<<" error="<<bp.error<<endl;
333.    }
334.    cout<<"trainning complete..."<<endl;
335.    double m[innode]={1,1,1};
336.    double *r=bp.recognize(m);
337.    for(int i=0;i<outnode;++i)
338.       cout<<bp.result[i]<<" ";
339.    double cha[trainsample][outnode];
340.    double mi=100;
341.    double index;
342.    for(int i=0;i<trainsample;i++)
343.    {
344.        for(int j=0;j<outnode;j++)
345.        {
346.            //�Ҳ�ֵ��С���Ǹ�����
347.            cha[i][j]=(double)(fabs(Y[i][j]-bp.result[j]));
348.            if(cha[i][j]<mi)
349.            {
350.                mi=cha[i][j];
351.                index=i;
352.            }
353.        }
354.    }
355.    for(int i=0;i<innode;++i)
356.       cout<<m[i];
357.    cout<<" is "<<index<<endl;
358.    cout<<endl;
359.    return 0;
360.}

*/
