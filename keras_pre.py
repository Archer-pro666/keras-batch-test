import matplotlib
matplotlib.use('Agg')
import os
from keras.models import load_model
import numpy as np
import cv2

#加载模型h5文件
model = load_model("D:/keras/模型/cell.h5")
model.summary()

#规范化图片大小和像素值
def get_inputs(src=[]):
    pre_x = []
    for s in src:
        input = cv2.imread(s)
        input = cv2.resize(input, (150, 150))
        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        pre_x.append(input)  # input一张图片
    pre_x = np.array(pre_x) / 255.0
    return pre_x

#要预测的图片保存在这里

predict_dir = 'D:/validation222/'
test11 = os.listdir(predict_dir)

images = []#新建一个列表保存预测图片的地址

#调用函数，规范化图片
c_c=0;c_n=0;c_r=0;c_w=0;n_c=0;n_n=0;n_r=0;n_w=0;r_c=0;r_n=0;r_r=0;r_w=0;w_c=0;w_n=0;w_r=0;w_w=0

def test(begin,boost,LABEL):
    global c_c,c_n,c_r,c_w,n_c,n_n,n_r,n_w,r_c,r_n,r_r,r_w,w_c,w_n,w_r,w_w
    boost=int(boost);begin=int(begin);LABEL=int(LABEL)
    for testpath in test11:
        for fn in os.listdir(os.path.join(predict_dir, testpath)):
            if fn.endswith('bmp'):
                fd = os.path.join(predict_dir, testpath, fn)
                images.append(fd)

    for i in range(boost):
        pre_x = get_inputs(images[begin:begin+boost])
        pre_y = model.predict(pre_x)
        pre_y=np.argmax(pre_y[i])
        print(pre_y)
        if LABEL==0:
            if pre_y ==0:
                c_c +=1
            elif pre_y ==1:
                c_n +=1
            elif pre_y ==2:
                c_r+=1
            elif pre_y ==3:
                c_w+=1
        elif LABEL==1:
            if pre_y ==0:
                n_c +=1
            elif pre_y ==1:
                n_n +=1
            elif pre_y ==2:
                n_r+=1
            elif pre_y ==3:
                n_w+=1
        elif LABEL==2:
            if pre_y ==0:
                r_c +=1
            elif pre_y ==1:
                r_n +=1
            elif pre_y ==2:
                r_r+=1
            elif pre_y ==3:
                r_w+=1
        elif LABEL==3:
            if pre_y ==0:
                w_c +=1
            elif pre_y ==1:
                w_n +=1
            elif pre_y ==2:
                w_r+=1
            elif pre_y ==3:
                w_w+=1
        result1txt = str(pre_y)      # pre_y是预测标签，先将其转为字符串才能写入
        result2txt = str(LABEL)      #LABEL是输入的真实标签，也转为字符串
        with open('D:/keras/文本/test.txt','a') as file_handle:
           file_handle.write(result2txt)     # 写入真实标签
           file_handle.write('\n')         # 数据自动转行，否则会覆盖上一条数据
        with open('D:/keras/文本/pred.txt','a') as file_handle1:
           file_handle1.write(result1txt)     # 写入预测标签
           file_handle1.write('\n')         # 数据自动转行，否则

if __name__ == '__main__':
    sum_caot=1087;sum_nse=940;sum_rbc=10687;sum_wbc=6747
    i=10;s=0;L=int(test11[0]);a=sum_caot%i;b=sum_nse%i;c=sum_rbc%i;d=sum_wbc%i
    for w in range(int((sum_caot-a)/i+1)):
        if s+i<=sum_caot-a:
            test(s,i,L)
            s+=i
            print('已测CAOT数目：',s)
        elif s<=sum_caot:
            test(s,a,L)
            s+=a
            print('已测CAOT数目：',s)

    for w in range(int((sum_nse-b)/i+1)):
        if s+i-sum_caot<=sum_nse-b:
            L=int(test11[1])
            test(s,i,L)
            s+=i
            print('已测NSE数目：',s-sum_caot)
        elif s-sum_caot<=sum_nse:
            test(s,b,L)
            s+=b
            print('已测NSE数目：',s-sum_caot)

    for w in range(int((sum_rbc-c)/i+1)):
        if s+i-sum_caot-sum_nse<=sum_rbc-c:
            L=int(test11[2])
            test(s,i,L)
            s+=i
            print('已测RBC数目：',s-sum_caot-sum_nse)
        elif s-sum_caot-sum_nse<=sum_rbc:
            test(s,c,L)
            s+=c
            print('已测RBC数目：',s-sum_caot-sum_nse)

    for w in range(int((sum_wbc-d)/i+1)):
        if s+i-sum_caot-sum_nse-sum_rbc<=sum_wbc-d:
            L=int(test11[3])
            test(s,i,L)
            s+=i
            print('已测WBC数目：',s-sum_caot-sum_nse-sum_rbc)
        elif s-sum_caot-sum_nse-sum_rbc<=sum_wbc:
            test(s,d,L)
            s+=d
            print('已测WBC数目：',s-sum_caot-sum_nse-sum_rbc)

    CAOT=c_c+n_c+r_c+w_c
    NSE=c_n+n_n+r_n+w_n
    RBC=c_r+n_r+r_r+w_r
    WBC=c_w+n_w+r_w+w_w

    ALL=CAOT+NSE+RBC+WBC

    print('CAOT预测总数：',CAOT)
    print('NSE预测总数：',NSE)
    print('RBC预测总数：',RBC)
    print('WBC预测总数：',WBC)
    print('已经检测细胞数：',ALL)

    c_o=c_n+c_r+c_w
    n_o=n_c+n_r+n_w
    r_o=r_c+r_n+r_w
    w_o=w_c+w_n+w_r

    o_c=n_c+r_c+w_c
    o_n=c_n+r_n+w_n
    o_r=c_r+n_r+w_r
    o_w=c_w+n_w+r_w

    try:
        pre_c=c_c/(c_c+o_c)
        pre_n=n_n/(n_n+o_n)
        pre_r=r_r/(r_r+o_r)
        pre_w=w_w/(w_w+o_w)
        pre_c = str(pre_c*100) + '%'   #转化为百分数
        pre_n = str(pre_n*100) + '%'   #转化为百分数
        pre_r = str(pre_r*100) + '%'   #转化为百分数
        pre_w = str(pre_w*100) + '%'   #转化为百分数

        recall_c=c_c/(c_c+c_o)
        recall_n=n_n/(n_n+n_o)
        recall_r=r_r/(r_r+r_o)
        recall_w=w_w/(w_w+w_o)
        recall_c = str(recall_c*100) + '%'   #转化为百分数
        recall_n = str(recall_n*100) + '%'   #转化为百分数
        recall_r = str(recall_r*100) + '%'   #转化为百分数
        recall_w = str(recall_w*100) + '%'   #转化为百分数

        print("CAOT精准率：",pre_c)
        print("NSE精准率：",pre_n)
        print("RBC精准率：",pre_r)
        print("WBC精准率：",pre_w)

        print("CAOT召回率：",recall_c)
        print("NSE召回率：",recall_n)
        print("RBC召回率：",recall_r)
        print("WBC召回率：",recall_w)
    except:
        pass

    if ALL<=19461:
          #打印计数结果：
          print('CAOT判为CAOT： %(1)d,CAOT判为NSE： %(2)d,CAOT判为RBC： %(3)d,CAOT判为WBC： %(4)d,'%{'1':c_c,'2':c_n,'3':c_r,'4':c_w})
          print('NSE判为CAOT：  %(1)d,NSE判为NSE：  %(2)d,NSE判为RBC：  %(3)d,NSE判为WBC：  %(4)d,'%{'1':n_c,'2':n_n,'3':n_r,'4':n_w})
          print('RBC判为CAOT：  %(1)d,RBC判为NSE：  %(2)d,RBC判为RBC：  %(3)d,RBC判为WBC：  %(4)d,'%{'1':r_c,'2':r_n,'3':r_r,'4':r_w})
          print('WBC判为CAOT：  %(1)d,WBC判为NSE：  %(2)d,WBC判为RBC：  %(3)d,WBC判为WBC：  %(4)d,'%{'1':w_c,'2':w_n,'3':w_r,'4':w_w})

          print('其他细胞判为椭圆形草酸钙的数目：',o_c)
          print('其他细胞判为非鳞状上皮细胞的数目：',o_n)
          print('其他细胞判为红细胞的数目：',o_r)
          print('其他细胞判为白细胞的数目：',o_w)

          print('椭圆形草酸钙判为其他的数目：',c_o)
          print('非鳞状上皮细胞为其他的数目：',n_o)
          print('红细胞判为其他的数目：',r_o)
          print('白细胞判为其他的数目：',w_o)
    else:
          print ("end")