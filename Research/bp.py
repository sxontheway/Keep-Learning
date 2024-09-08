import os
import struct
import numpy as np
import matplotlib.pyplot as plt
import time

# dataset download link http://yann.lecun.com/exdb/mnist/


def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels.idx1-ubyte'
                               %kind)
    images_path = os.path.join(path,
                               '%s-images.idx3-ubyte'
                               %kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)
    
    #处理标签成（60000，10）形状
    new_labels=[]
    for i in labels:
        if   i==0:
            new_labels.append([1,0,0,0,0,0,0,0,0,0])
        elif i==1:
            new_labels.append([0,1,0,0,0,0,0,0,0,0])
        elif i==2:
            new_labels.append([0,0,1,0,0,0,0,0,0,0])
        elif i==3:
            new_labels.append([0,0,0,1,0,0,0,0,0,0])
        elif i==4:
            new_labels.append([0,0,0,0,1,0,0,0,0,0])
        elif i==5:
            new_labels.append([0,0,0,0,0,1,0,0,0,0])
        elif i==6:
            new_labels.append([0,0,0,0,0,0,1,0,0,0])
        elif i==7:
            new_labels.append([0,0,0,0,0,0,0,1,0,0])
        elif i==8:
            new_labels.append([0,0,0,0,0,0,0,0,1,0])
        else:
            new_labels.append([0,0,0,0,0,0,0,0,0,1])
    #输出在0-1之间
    return images/255, np.array(new_labels)

def sigmoid(z, first_derivative=False):
    x=1.0/(1.0+np.exp(-z))
    if first_derivative:
        return x*(1.0-x)
    else:
        return x

def verify_validity(x_data,y_data,N):
    newh=[]
    newz=[]
    for j in range(k):
        if j==0:
            newz.append(np.matmul(x_data[:N],w[j])+np.matmul(np.ones(shape=(N,1)),b[j]))
            newh.append(sigmoid(newz[j])) 
        else:
            newz.append(np.matmul(newh[j-1],w[j])+np.matmul(np.ones(shape=(N,1)),b[j]))
            newh.append(sigmoid(newz[j]))
    for j in range(k):
        if j==0:
            newz[j]=np.matmul(x_data[:N],w[j])+np.matmul(np.ones(shape=(N,1)),b[j])
            newh[j]=sigmoid(newz[j])
        else:
            newz[j]=np.matmul(newh[j-1],w[j])+np.matmul(np.ones(shape=(N,1)),b[j])
            newh[j]=sigmoid(newz[j])
            
    y_predict = np.argmax(newh[k-1], axis=1)
    # print("y_predict:",y_predict)
    y_actual = np.argmax(y_data[:N], axis=1)
    accuracy = np.sum(np.equal(y_predict,y_actual))/len(y_actual)
    #损失函数：
    loss=np.square(y_data[:N]-newh[k-1]).sum()/(2*N)
    #训练过程损失
    return loss,accuracy

def visualize_result(save_path,accuracies,losses):
    # if not os.path.exists(save_path):
    #     os.mkdir(r'%s' % save_path)
    # Accurary_name="Accuracy_input_dim_%d-hidden_dim_%d-output_dim_%d-num_epochs_%d-N_train_%d-N_test_%d-learning_rate_%f.png"%(input_dim,hidden_dim,output_dim,
    #                                                                                                         num_epochs,N_train,N_test,learning_rate)
    # Loss_name="Loss_input_dim_%d-hidden_dim_%d-output_dim_%d-num_epochs_%d-N_train_%d-N_test_%d-learning_rate_%f.png"%(input_dim,hidden_dim,output_dim,
    #                                                                                                     num_epochs,N_train,N_test,learning_rate)
    #准确率画图
    plt.plot(accuracies[:,0],accuracies[:,1],label='Accuracy_train')
    plt.plot(accuracies[:,0],accuracies[:,2],label='Accuracy_test')
    # plt.title("Accuracy_input_dim=%d,hidden_dim=%d,output_dim=%d,\n num_epochs=%d,N_train=%d,N_test=%d,learning_rate=%f"%(input_dim,hidden_dim,output_dim,
    #                                                                                                     num_epochs,N_train,N_test,learning_rate))
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.show()
    # plt.savefig(os.path.join(save_path,Accurary_name), dpi=300)
    # plt.close("all")
    #损失函数画图
    plt.plot(losses[:,0],losses[:,1],label='Loss_train')
    plt.plot(losses[:,0],losses[:,2],label='Loss_test')
    # plt.title("Loss_input_dim=%d,hidden_dim=%d,output_dim=%d,\n num_epochs=%d,N_train=%d,N_test=%d,learning_rate=%f"%(input_dim,hidden_dim,output_dim,
    #                                                                                                     num_epochs,N_train,N_test,learning_rate))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.show()

def train():
    #初始化w、b
    for j in range(k):
        w.append(( 2*np.random.random((net_dim[j],net_dim[j+1])) - 1 )/1)
        b.append((2*np.random.random((1,net_dim[j+1])) - 1 )/1)
    #初始化z、h
    for j in range(k):
        if j==0:
            z.append(np.matmul(x_train[:min_batch],w[j])+np.matmul(np.ones(shape=(min_batch,1)),b[j]))
            h.append(sigmoid(z[j])) 
        else:
            z.append(np.matmul(h[j-1],w[j])+np.matmul(np.ones(shape=(min_batch,1)),b[j]))
            h.append(sigmoid(z[j]))
    #初始化残差delta  
    for j in range(k):
        if j==0:
            delta.append(-(y_train[:min_batch]-h[k-1])*sigmoid(z[k-1], first_derivative=True))
        else:
            delta.append(np.matmul(delta[j-1],w[k-j].T)*sigmoid(z[k-1-j], first_derivative=True))
    #计算loss
    for i in range(num_epochs):
        for num_bat in range(int(N_train/min_batch)):
            #前向传播：
            for j in range(k):
                if j==0:
                    z[j]=np.matmul(x_train[num_bat*min_batch:(num_bat+1)*min_batch],w[j])+np.matmul(np.ones(shape=(min_batch,1)),b[j])
                    h[j]=sigmoid(z[j])
                else:
                    z[j]=np.matmul(h[j-1],w[j])+np.matmul(np.ones(shape=(min_batch,1)),b[j])
                    h[j]=sigmoid(z[j])
            #损失函数：
            L=np.square(y_train[num_bat*min_batch:(num_bat+1)*min_batch]-h[k-1]).sum()/(2*min_batch)
            #反向传播：
            #计算残差:
            for j in range(k):
                if j==0:
                    delta[j]=-(y_train[num_bat*min_batch:(num_bat+1)*min_batch]-h[k-1])*sigmoid(z[k-1], first_derivative=True)
                else:
                    delta[j]=np.matmul(delta[j-1],w[k-j].T)*sigmoid(z[k-1-j], first_derivative=True)
            #权重和偏置更新:
            for j in range(k):
                if j==0:
                    w[0] += -learning_rate*np.matmul(x_train[num_bat*min_batch:(num_bat+1)*min_batch].T,delta[k-1])/min_batch
                    b[0] += -learning_rate*np.matmul(np.ones(shape=(min_batch,1)).T,delta[k-1])/min_batch
                else:
                    w[j] += -learning_rate*np.matmul(h[j-1].T,delta[k-1-j])/min_batch
                    b[j] += -learning_rate*np.matmul(np.ones(shape=(min_batch,1)).T,delta[k-1-j])/min_batch
         # 记录结果
        if True:
            #训练集准确度
            loss_train,accuracy_train = verify_validity(x_train,y_train,N_train)
            #测试集准确度
            loss_test,accuracy_test = verify_validity(x_test,y_test,N_test)
            #训练过程损失
            losses.append([i,loss_train,loss_test])
            accuracies.append([i,accuracy_train,accuracy_test])
        if i%1==0:
            print('Epoch: %d Loss:%f Loss_train:%f Loss_test:%f Accuracy_train: %f Accuracy_test: %f' %(i,L,loss_train,loss_test,accuracy_train,accuracy_test))
                
    return np.array(losses),np.array(accuracies)
      
if __name__ == '__main__':    
    #读取数据     
    folder_path='D:\\Code\\Nju_study\\'
    x_train,y_train=load_mnist(folder_path+'mnist','train') #(60000,input_dim),(60000,output_dim)
    x_test,y_test=load_mnist(folder_path+'mnist','t10k')    #(10000,input_dim),(10000,output_dim)
    #定义参数
    net_dim=[784,128,10]
    num_epochs = 50
    learning_rate= 1
    N_train=60000
    min_batch= 100
    N_test=10000
    k=len(net_dim)-1
    #定义变量
    losses = []
    accuracies=[]
    w=[]
    b=[]
    z=[]
    h=[]
    delta=[]  
    #开始训练
    print("==============================================================================================================================")
    start_time = time.time()                                          #训练开始时间
    losses,accuracies=train()
    end_time = time.time()                                            #训练结束时间
    run_time=end_time-start_time                                      #训练时间，单位为秒
    print("本次运行时间:%d h %d m %d s"%(run_time//3600,(run_time-run_time//3600*3600)//60,run_time%60))  
    #输出结果 
    visualize_result(folder_path+'learning_rate',accuracies,losses) 
