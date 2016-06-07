//将三位二进制数转为一位十进制数
/*
03.#include<iostream>
04.#include<cmath>
05.using namespace std;
06.
07.#define  innode 3  //输入结点数
08.#define  hidenode 10//隐含结点数
09.#define  outnode 1 //输出结点数
10.#define  trainsample 8//BP训练样本数
11.class BpNet
12.{
13.public:
14.    void train(double p[trainsample][innode ],double t[trainsample][outnode]);//Bp训练
15.    double p[trainsample][innode];     //输入的样本
16.    double t[trainsample][outnode];    //样本要输出的
17.
18.    double *recognize(double *p);//Bp识别
19.
20.    void writetrain(); //写训练完的权值
21.    void readtrain(); //读训练好的权值，这使的不用每次去训练了，只要把训练最好的权值存下来就OK
22.
23.    BpNet();
24.    virtual ~BpNet();
25.
26.public:
27.    void init();
28.    double w[innode][hidenode];//隐含结点权值
29.    double w1[hidenode][outnode];//输出结点权值
30.    double b1[hidenode];//隐含结点阀值
31.    double b2[outnode];//输出结点阀值
32.
33.    double rate_w; //权值学习率（输入层-隐含层)
34.    double rate_w1;//权值学习率 (隐含层-输出层)
35.    double rate_b1;//隐含层阀值学习率
36.    double rate_b2;//输出层阀值学习率
37.
38.    double e;//误差计算
39.    double error;//允许的最大误差
40.    double result[outnode];// Bp输出
41.};
42.
43.BpNet::BpNet()
44.{
45.    error=1.0;
46.    e=0.0;
47.
48.    rate_w=0.9;  //权值学习率（输入层--隐含层)
49.    rate_w1=0.9; //权值学习率 (隐含层--输出层)
50.    rate_b1=0.9; //隐含层阀值学习率
51.    rate_b2=0.9; //输出层阀值学习率
52.}
53.
54.BpNet::~BpNet()
55.{
56.
57.}
58.
59.void winit(double w[],int n) //权值初始化
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
75.    double pp[hidenode];//隐含结点的校正误差
76.    double qq[outnode];//希望输出值与实际输出值的偏差
77.    double yd[outnode];//希望输出值
78.
79.    double x[innode]; //输入向量
80.    double x1[hidenode];//隐含结点状态值
81.    double x2[outnode];//输出结点状态值
82.    double o1[hidenode];//隐含层激活值
83.    double o2[hidenode];//输出层激活值
84.
85.    for(int isamp=0;isamp<trainsample;isamp++)//循环训练一次样品
86.    {
87.        for(int i=0;i<innode;i++)
88.            x[i]=p[isamp][i]; //输入的样本
89.        for(int i=0;i<outnode;i++)
90.            yd[i]=t[isamp][i]; //期望输出的样本
91.
92.        //构造每个样品的输入和输出标准
93.        for(int j=0;j<hidenode;j++)
94.        {
95.            o1[j]=0.0;
96.            for(int i=0;i<innode;i++)
97.                o1[j]=o1[j]+w[i][j]*x[i];//隐含层各单元输入激活值
98.            x1[j]=1.0/(1+exp(-o1[j]-b1[j]));//隐含层各单元的输出
99.            //    if(o1[j]+b1[j]>0) x1[j]=1;
100.            //else x1[j]=0;
101.        }
102.
103.        for(int k=0;k<outnode;k++)
104.        {
105.            o2[k]=0.0;
106.            for(int j=0;j<hidenode;j++)
107.                o2[k]=o2[k]+w1[j][k]*x1[j]; //输出层各单元输入激活值
108.            x2[k]=1.0/(1.0+exp(-o2[k]-b2[k])); //输出层各单元输出
109.            //    if(o2[k]+b2[k]>0) x2[k]=1;
110.            //    else x2[k]=0;
111.        }
112.
113.        for(int k=0;k<outnode;k++)
114.        {
115.            qq[k]=(yd[k]-x2[k])*x2[k]*(1-x2[k]); //希望输出与实际输出的偏差
116.            for(int j=0;j<hidenode;j++)
117.                w1[j][k]+=rate_w1*qq[k]*x1[j];  //下一次的隐含层和输出层之间的新连接权
118.        }
119.
120.        for(int j=0;j<hidenode;j++)
121.        {
122.            pp[j]=0.0;
123.            for(int k=0;k<outnode;k++)
124.                pp[j]=pp[j]+qq[k]*w1[j][k];
125.            pp[j]=pp[j]*x1[j]*(1-x1[j]); //隐含层的校正误差
126.
127.            for(int i=0;i<innode;i++)
128.                w[i][j]+=rate_w*pp[j]*x[i]; //下一次的输入层和隐含层之间的新连接权
129.        }
130.
131.        for(int k=0;k<outnode;k++)
132.        {
133.            e+=fabs(yd[k]-x2[k])*fabs(yd[k]-x2[k]); //计算均方差
134.        }
135.        error=e/2.0;
136.
137.        for(int k=0;k<outnode;k++)
138.            b2[k]=b2[k]+rate_b2*qq[k]; //下一次的隐含层和输出层之间的新阈值
139.        for(int j=0;j<hidenode;j++)
140.            b1[j]=b1[j]+rate_b1*pp[j]; //下一次的输入层和隐含层之间的新阈值
141.    }
142.}
143.
144.double *BpNet::recognize(double *p)
145.{
146.    double x[innode]; //输入向量
147.    double x1[hidenode]; //隐含结点状态值
148.    double x2[outnode]; //输出结点状态值
149.    double o1[hidenode]; //隐含层激活值
150.    double o2[hidenode]; //输出层激活值
151.
152.    for(int i=0;i<innode;i++)
153.        x[i]=p[i];
154.
155.    for(int j=0;j<hidenode;j++)
156.    {
157.        o1[j]=0.0;
158.        for(int i=0;i<innode;i++)
159.            o1[j]=o1[j]+w[i][j]*x[i]; //隐含层各单元激活值
160.        x1[j]=1.0/(1.0+exp(-o1[j]-b1[j])); //隐含层各单元输出
161.        //if(o1[j]+b1[j]>0) x1[j]=1;
162.        //    else x1[j]=0;
163.    }
164.
165.    for(int k=0;k<outnode;k++)
166.    {
167.        o2[k]=0.0;
168.        for(int j=0;j<hidenode;j++)
169.            o2[k]=o2[k]+w1[j][k]*x1[j];//输出层各单元激活值
170.        x2[k]=1.0/(1.0+exp(-o2[k]-b2[k]));//输出层各单元输出
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
189.    //隐含结点权值写入
190.    if(( stream0 = fopen("w.txt", "w+" ))==NULL)
191.    {
192.        cout<<"创建文件失败!";
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
204.    //输出结点权值写入
205.    if(( stream1 = fopen("w1.txt", "w+" ))==NULL)
206.    {
207.        cout<<"创建文件失败!";
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
219.    //隐含结点阀值写入
220.    if(( stream2 = fopen("b1.txt", "w+" ))==NULL)
221.    {
222.        cout<<"创建文件失败!";
223.        exit(1);
224.    }
225.    for(i=0;i<hidenode;i++)
226.        fprintf(stream2, "%f\n",b1[i]);
227.    fclose(stream2);
228.
229.    //输出结点阀值写入
230.    if(( stream3 = fopen("b2.txt", "w+" ))==NULL)
231.    {
232.        cout<<"创建文件失败!";
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
249.    //隐含结点权值读出
250.    if(( stream0 = fopen("w.txt", "r" ))==NULL)
251.    {
252.        cout<<"打开文件失败!";
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
266.    //输出结点权值读出
267.    if(( stream1 = fopen("w1.txt", "r" ))==NULL)
268.    {
269.        cout<<"打开文件失败!";
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
283.    //隐含结点阀值读出
284.    if(( stream2 = fopen("b1.txt", "r" ))==NULL)
285.    {
286.        cout<<"打开文件失败!";
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
297.    //输出结点阀值读出
298.    if(( stream3 = fopen("b2.txt", "r" ))==NULL)
299.    {
300.        cout<<"打开文件失败!";
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
313.//输入样本
314.double X[trainsample][innode]= {
315.    {0,0,0},{0,0,1},{0,1,0},{0,1,1},{1,0,0},{1,0,1},{1,1,0},{1,1,1}
316.    };
317.//期望输出样本
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
346.            //找差值最小的那个样本
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
