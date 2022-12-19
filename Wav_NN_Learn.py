import numpy as np
import tensorflow as tf
import random as rn
import os
os.environ['PYTHONHASHSEED'] = '0'

np.random.seed(42)
rn.seed(12345)
from keras.layers import Convolution2D, BatchNormalization, MaxPooling2D, Dense, Input, Dropout, Flatten
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.optimizers import Adam, Adagrad, SGD, RMSprop, Adadelta, Adamax, Nadam, Ftrl
import time
import pandas as pd
from keras.callbacks import TensorBoard
import os
from sklearn.model_selection import train_test_split
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from scipy import signal
from scipy.io import wavfile
import random






def Load_Wav_(WorkDir):
    Input_Files = []
    Source_Samples= []
    for d, dirs, files in os.walk(WorkDir):
        for file in files:
            Input_Files.append(file)
    GFile  = []
    for file in Input_Files:
            if file.endswith(".wav"):
                sample_rate, samples = wavfile.read(str(WorkDir)+ file)
                if sample_rate != 8000:
                    continue
                if max(abs(samples)) < 410:
                    continue
                if len(samples) < int(0.1 * sample_rate):
                    continue
                GFile.append(file)
#                 samples = Convert_To_06(samples)
                Source_Samples.append(samples)

    return  Source_Samples, GFile



def Add_Zero(specgram, TargetColumnNumber, StartSignalPosition):

    if len(specgram[0]) >= TargetColumnNumber:
         return specgram

    full_array = np.zeros((len(specgram), TargetColumnNumber))
    full_array[:, :len(specgram[0])-TargetColumnNumber] = specgram
    if StartSignalPosition == 0:
        full_array[:, :len(specgram[0]) - TargetColumnNumber] = specgram
    elif StartSignalPosition == TargetColumnNumber - len(specgram[0]):
        full_array[:, StartSignalPosition:] = specgram
    else:
        full_array[:, StartSignalPosition:StartSignalPosition+len(specgram[0]) - TargetColumnNumber] = specgram

    return full_array
def log_specgram(audio, window_size, sample_rate=8000,
                 eps=1e-10, windoe_fuction='hann'):
    nperseg = int(round(window_size * sample_rate / 1000))
    noverlap = int(round(window_size / 2 * sample_rate / 1000))
    freqs, times, spec = signal.spectrogram(audio,
                                            fs=sample_rate,
                                            window=windoe_fuction,
                                            nperseg=nperseg,
                                            noverlap=noverlap,
                                            detrend=False)
    return freqs, times, np.log(spec.astype(np.float32) + eps)
def Convert_Wav_To_specgram(SamplesList, Input_Files, window_size, windoe_fuction):
    samples = np.zeros(int(0.6 * 8000))
    _, _, specgram = log_specgram(audio=samples, window_size=window_size, windoe_fuction=windoe_fuction)
    TargetColumnNumber = len(specgram[0])
    TargetRow = len(specgram)
    x = []
    y = []
    for i in range(len(SamplesList)):
        _, _, specgram = log_specgram(audio=SamplesList[i], window_size=window_size, windoe_fuction=windoe_fuction)
        specgram = Add_Zero(specgram, TargetColumnNumber, 0)
        x.append(specgram)
        file = Input_Files[i]
        if 'cl_1' in file:
            y.append([0, 0, 1])
        elif 'cl_2' in file:
            y.append([1, 0, 0])
        else:
            y.append([0, 1, 0])
        
 
    x = np.array(x)
    x = x.reshape(tuple(list(x.shape) + [1]))
    y = np.array(y)
    return x, y
def RandomozeArrays(SourceArrayX, SourceArrayY):
    TargetArrayX=[]
    TargetArrayY=[]
    while 0 < len(SourceArrayX):
        Index = random.randint(0, len(SourceArrayX) - 1)
        TargetArrayX.append(SourceArrayX[Index])
        del SourceArrayX[Index]
        TargetArrayY.append(SourceArrayY[Index])
        del SourceArrayY[Index]      
        
    return TargetArrayX,TargetArrayY






# def Learn_NN_5L_(TrainDir,ValidDir, RezDir,NN_Name,Epochs=30, window_size=25, windoe_fuction='hann'):
#     Source_Samples, Input_Files= Load_Wav_(TrainDir)
#     print('end load train data', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
#     Source_Samples, Input_Files = RandomozeArrays(Source_Samples, Input_Files)
#     print('end Randomize train data', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    
    
#     X_Train, Y_Train = Convert_Wav_To_specgram(SamplesList=Source_Samples, Input_Files=Input_Files,
#                                            window_size=window_size, windoe_fuction=windoe_fuction)
#     print('end convert train data', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
     
#     Source_Samples, Input_Files = Load_Wav_(ValidDir)
#     print('end load valid data', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

#     X_val1, y_val1 = Convert_Wav_To_specgram(SamplesList=Source_Samples, Input_Files=Input_Files,
#                                                      window_size=window_size, windoe_fuction=windoe_fuction)
#     print('end convert valid data', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

#     input_shape = (X_Train.shape[1], X_Train.shape[2], 1)
#     model = Sequential()

#     model.add(BatchNormalization(input_shape = input_shape))
#     model.add(Convolution2D(48, (5, 5), strides = (3, 3), padding = 'same',input_shape = input_shape))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding = 'same'))
#     model.add(BatchNormalization())
#     model.add(Convolution2D(40, (3, 3), strides = (2, 2), padding = 'same'))
#     model.add(Activation('relu'))
#     model.add(Flatten())
# #     model.add(Dense(45))
# #     model.add(Activation('relu'))
# #     model.add(Dense(30))
# #     model.add(Activation('relu'))
#     model.add(Dense(15))
#     model.add(Activation('relu'))
#     model.add(Dense(10))
#     model.add(Activation('relu'))
#     model.add(Dense(3))
#     model.add(Activation('softmax'))

#     model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
#     csv_logger = CSVLogger(RezDir+NN_Name+'_training__log.csv', separator=',', append=False)

    
#     checkpoint = ModelCheckpoint(filepath=RezDir+NN_Name+'_Best.hdf5',
#                  monitor='val_accuracy',
#                  save_best_only=True,
#                  mode='max',
#                  verbose=1)
#     model.fit(X_Train, Y_Train,
#           batch_size = 64,
#           epochs = Epochs,shuffle=True,
#           validation_data=(X_val1, y_val1),
#           callbacks=[checkpoint, csv_logger])
#     model.save(filepath=RezDir+NN_Name+'_Final.hdf5')
    
    
    
    
    
    
    
def Learn_NN_5L_Custom(TrainDir,ValidDir, RezDir,NN_Name, Neurons=40, Epochs=30, window_size=25, windoe_fuction='hann'):
    Source_Samples, Input_Files= Load_Wav_(TrainDir)
    print('end load train data', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    Source_Samples, Input_Files = RandomozeArrays(Source_Samples, Input_Files)
    print('end Randomize train data', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    
    
    X_Train, Y_Train = Convert_Wav_To_specgram(SamplesList=Source_Samples, Input_Files=Input_Files,
                                           window_size=window_size, windoe_fuction=windoe_fuction)
    print('end convert train data', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
     
    Source_Samples, Input_Files = Load_Wav_(ValidDir)
    print('end load valid data', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    X_val1, y_val1 = Convert_Wav_To_specgram(SamplesList=Source_Samples, Input_Files=Input_Files,
                                                     window_size=window_size, windoe_fuction=windoe_fuction)
    print('end convert valid data', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    input_shape = (X_Train.shape[1], X_Train.shape[2], 1)
    model = Sequential()

    model.add(BatchNormalization(input_shape = input_shape))
    model.add(Convolution2D(48, (5, 5), strides = (3, 3), padding = 'same',input_shape = input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding = 'same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(40, (3, 3), strides = (2, 2), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Flatten())
#     model.add(Dense(45))
#     model.add(Activation('relu'))
#     model.add(Dense(15))
#     model.add(Activation('relu'))
#     model.add(Dense(10))
#     model.add(Activation('relu'))
    model.add(Dense(Neurons))
    model.add(Activation('relu'))
    model.add(Dense(3))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    csv_logger = CSVLogger(RezDir+NN_Name+'_training__log.csv', separator=',', append=False)

    
    checkpoint = ModelCheckpoint(filepath=RezDir+NN_Name+'_Best.hdf5',
                 monitor='val_accuracy',
                 save_best_only=True,
                 mode='max',
                 verbose=1)
    model.fit(X_Train, Y_Train,
          batch_size = 64,
          epochs = Epochs,shuffle=True,
          validation_data=(X_val1, y_val1),
          callbacks=[checkpoint, csv_logger])
    model.save(filepath=RezDir+NN_Name+'_Final.hdf5')
    
    
    
    



def TestNN_(NetName, SourceDir, TargetFile, window_size):
    Input_Files = []
    Source_Samples = []
    for d, dirs, files in os.walk(SourceDir):
        for file in files:
            if file.endswith(".wav"):
                sample_rate, samples = wavfile.read(SourceDir + file)
                if sample_rate != 8000:
                     continue
                if max(abs(samples)) < 410:
                    continue
                if len(samples) < int(0.1 * sample_rate):
                    continue
                Input_Files.append(file)
#                 samples = Convert_To_06(samples)
                Source_Samples.append(samples)

    x, y = Convert_Wav_To_specgram(SamplesList=Source_Samples, Input_Files=Input_Files,
                                             window_size=window_size, windoe_fuction='hann')

    new_model = load_model(NetName)
    pred = new_model.predict(x)
    f = open(TargetFile+'_FilesReport.csv', 'w', newline='\n')
    f.write('NetName = %s, Files %s \n'%(NetName,SourceDir))
    f.write('File Name,Marked As,Recognized As,Cl 1,Cl 2,Cl 3, \n')
    CodeList = ['Cl 2', 'Cl 3', 'Cl 1']
    SemplCount= [0,0,0]
    StatRez = [[0,0,0],[0,0,0],[0,0,0]]
    for i in range(len(pred)):
        YY = list(y[i])
        Rez = list(pred[i])
        TrueCalss = YY.index(max(YY))
        NNClass = Rez.index(max(Rez))
        SemplCount[TrueCalss] +=1
        StatRez[TrueCalss][NNClass] += 1
        f.write( '%s,%s,%s, %f , %f, %f,\n' % (Input_Files[i], CodeList[TrueCalss], CodeList[NNClass], pred[i][2], pred[i][0], pred[i][1]))

    f.close()
    f = open(TargetFile+'_Report.csv', 'w', newline='\n')
    f.write('NetName = %s, Files %s \n'%(NetName,SourceDir))
    f.write('Var,Cl 1,Cl 2,Cl 3, \n')

    f.write(  'Count,%s,%s, %s ,\n' % (SemplCount[2],SemplCount[0], SemplCount[1]))
    f.write('Cl 1 As,%s,%s, %s ,\n' % (StatRez[2][2], StatRez[2][0], StatRez[2][1]))
    f.write('Cl 2 As,%s,%s, %s ,\n' % (StatRez[0][2], StatRez[0][0], StatRez[0][1]))
    f.write('Cl 3 As,%s,%s, %s ,\n' % (StatRez[1][2], StatRez[1][0], StatRez[1][1]))
    trueclass =100.0* (StatRez[2][2] + StatRez[0][0] + StatRez[1][1])/ float( sum(SemplCount))
    
    for i in range(len(SemplCount)):
        for k in range(len(SemplCount)):
            if SemplCount[i] > 0:
                StatRez[i][k]=100.0*float(StatRez[i][k])/SemplCount[i]
    f.write('Cl 1 As %%,%.3f%%,%.3f%%, %.3f%% ,\n' % (StatRez[2][2], StatRez[2][0], StatRez[2][1]))
    f.write('Cl 2  As %%,%.3f%%,%.3f%%, %.3f%% ,\n' % (StatRez[0][2], StatRez[0][0], StatRez[0][1]))
    f.write('Cl 3 As %%,%.3f%%,%.3f%%, %.3f%% ,\n' % (StatRez[1][2], StatRez[1][0], StatRez[1][1]))
    
    f.write(',\n')
    
    f.write('Total acc. ,%.3f%% ,\n' % (trueclass))

    f.close()
    
    
    
    
    
    
def TestNN_Custom(NetName, SourceDir, TargetFolder, window_size):
    f = open(TargetFolder+'_Report.csv', 'a', newline='\n')
#     f.write('NetName = %s, Files %s, '%(NetName,SourceDir))
    f.write('NetName = %s'%(NetName))
    for dir in ('Train', 'Test', 'Valid'):
        Input_Files = []
        Source_Samples = []
        for d, dirs, files in os.walk(dir+"/"+SourceDir):
            for file in files:
                if file.endswith(".wav"):
                    sample_rate, samples = wavfile.read(dir+"/"+SourceDir + file)
                    if sample_rate != 8000:
                         continue
                    if max(abs(samples)) < 410:
                        continue
                    if len(samples) < int(0.1 * sample_rate):
                        continue
                    Input_Files.append(file)

                    Source_Samples.append(samples)

        x, y = Convert_Wav_To_specgram(SamplesList=Source_Samples, Input_Files=Input_Files,
                                                 window_size=window_size, windoe_fuction='hann')

        new_model = load_model(NetName)
        pred = new_model.predict(x)

        SemplCount= [0,0,0]
        StatRez = [[0,0,0],[0,0,0],[0,0,0]]
        for i in range(len(pred)):
            YY = list(y[i])
            Rez = list(pred[i])
            TrueCalss = YY.index(max(YY))
            NNClass = Rez.index(max(Rez))
            SemplCount[TrueCalss] +=1
            StatRez[TrueCalss][NNClass] += 1

        

        trueclass =100.0* (StatRez[2][2] + StatRez[0][0] + StatRez[1][1])/ float( sum(SemplCount))

        f.write('%.3f%%,' % (trueclass))
        
    f.write('\n')
    f.close()
    
    
    
    
    
    
    
# folder=f"Results15Neurons"
# folder = 'Lab4_experiment' + "/" + folder

# print('---start Learn---', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
# Learn_NN_5L_(TrainDir='/home/hlib/sem7/AI/NewData/Train/', 
#              ValidDir='/home/hlib/sem7/AI/NewData/Valid/', 
#              RezDir=f"/home/hlib/sem7/AI/{folder}/", 
#              NN_Name='NN_L5', Epochs=25, window_size=25, windoe_fuction='hann')

# print('---end  Learn---', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))





# for dir in ('Train', 'Test', 'Valid'):
#     TestNN_(NetName=f"/home/hlib/sem7/AI/{folder}/NN_L5_Best.hdf5",
#             SourceDir=f"/home/hlib/sem7/AI/NewData/{dir}/",
#             TargetFile=f"/home/hlib/sem7/AI/{folder}/NN_L5_rez_{dir}",
#             window_size=25)
    
    
    
    


def TestNN_Custom_2(NetName, SourceDir, TargetFile, window_size): 
    f = open(TargetFile+'Report.csv', 'a', newline='\n')
    
    Input_Files = []
    Source_Samples = []
    for d, dirs, files in os.walk(SourceDir):
        for file in files:
            if file.endswith(".wav"):
                sample_rate, samples = wavfile.read(SourceDir + file)
                if sample_rate != 8000:
                     continue
                if max(abs(samples)) < 410:
                    continue
                if len(samples) < int(0.1 * sample_rate):
                    continue
                Input_Files.append(file)
#           
                Source_Samples.append(samples)

    x, y = Convert_Wav_To_specgram(SamplesList=Source_Samples, Input_Files=Input_Files,
                                             window_size=window_size, windoe_fuction='hann')

    new_model = load_model(NetName)
    pred = new_model.predict(x)

    
    SemplCount= [0,0,0]
    StatRez = [[0,0,0],[0,0,0],[0,0,0]]
    for i in range(len(pred)):
        YY = list(y[i])
        Rez = list(pred[i])
        TrueCalss = YY.index(max(YY))
        NNClass = Rez.index(max(Rez))
        SemplCount[TrueCalss] +=1
        StatRez[TrueCalss][NNClass] += 1
   

   

 
    trueclass =100.0* (StatRez[2][2] + StatRez[0][0] + StatRez[1][1])/ float( sum(SemplCount))
    
    for i in range(len(SemplCount)):
        for k in range(len(SemplCount)):
            if SemplCount[i] > 0:
                StatRez[i][k]=100.0*float(StatRez[i][k])/SemplCount[i]

    f.write('%.3f%%,' % (trueclass))

    f.close()
    
    
    
    
    
    
    
# folder=f"Results15Neurons"
# folder = 'Lab4_experiment' + "/" + folder

# print('---start Learn---', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
# Learn_NN_5L_(TrainDir='/home/hlib/sem7/AI/NewData/Train/', 
#              ValidDir='/home/hlib/sem7/AI/NewData/Valid/', 
#              RezDir=f"/home/hlib/sem7/AI/{folder}/", 
#              NN_Name='NN_L5', Epochs=25, window_size=25, windoe_fuction='hann')

# print('---end  Learn---', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

# for dir in ('Train', 'Test', 'Valid'):
#     TestNN_(NetName=f"/home/hlib/sem7/AI/{folder}/NN_L5_Best.hdf5",
#             SourceDir=f"/home/hlib/sem7/AI/NewData/{dir}/",
#             TargetFile=f"/home/hlib/sem7/AI/{folder}/NN_L5_rez_{dir}",
#             window_size=25)
    
    
    
    
    
    
    
# d = 'Lab4_experiment'

# for neuro in [15]: 
# for neuro in range (5, 201, 5): 
# for neuro in range (205, 500, 5): 
#     folder=f"Results{neuro}Neurons"
#     folder = d + "/" + folder
    
#     e = neuro // 15

#     print(f"start train for {neuro} neurons")

#     Learn_NN_5L_Custom(TrainDir='/home/hlib/sem7/AI/NewData/Train/', 
#                  ValidDir='/home/hlib/sem7/AI/NewData/Valid/', 
#                  RezDir=f"/home/hlib/sem7/AI/{folder}/", 
#                  NN_Name='NN_L5', Neurons=neuro, Epochs=20+e, window_size=25, windoe_fuction='hann')


#     NetName = f"/home/hlib/sem7/AI/{folder}/NN_L5_Best.hdf5"
#     TargetFile = f"/home/hlib/sem7/AI/{d}/"

#     f = open(TargetFile+'Report.csv', 'a', newline='\n')
#     f.write('NetName = %s,'%(NetName))
#     f.close()
    
#     for dir in ('Train', 'Test', 'Valid'):
#         TestNN_Custom_2(NetName=NetName,
#                 SourceDir=f"/home/hlib/sem7/AI/NewData/{dir}/",
#                 TargetFile=TargetFile,
#                 window_size=25)
        
#     f = open(TargetFile+'Report.csv', 'a', newline='\n')
#     f.write('\n')
#     f.close()   
    
#     print(f"finish train for {neuro} neurons")
    
    
    
    
    
    
#     folder=f"Results15Neurons"
#     folder = 'Lab4_experiment' + "/" + folder

#     print('---start Learn---', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
#     Learn_NN_5L_(TrainDir='/home/hlib/sem7/AI/NewData/Train/', 
#                  ValidDir='/home/hlib/sem7/AI/NewData/Valid/', 
#                  RezDir=f"/home/hlib/sem7/AI/{folder}/", 
#                  NN_Name='NN_L5', Epochs=25, window_size=25, windoe_fuction='hann')

#     print('---end  Learn---', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

#     for dir in ('Train', 'Test', 'Valid'):
#         TestNN_(NetName=f"/home/hlib/sem7/AI/{folder}/NN_L5_Best.hdf5",
#                 SourceDir=f"/home/hlib/sem7/AI/NewData/{dir}/",
#                 TargetFile=f"/home/hlib/sem7/AI/{folder}/NN_L5_rez_{dir}",
#                 window_size=25)


#     Learn_NN_5L_(TrainDir='/home/hlib/sem7/AI/NewData/Train/', 
#                  ValidDir='/home/hlib/sem7/AI/NewData/Valid/', 
#                  RezDir=f"/home/hlib/sem7/AI/{folder}/", 
#                  NN_Name='NN_L5', Epochs=20+e, window_size=25, windoe_fuction='hann')


#     for dir in ('Train'):
#         TestNN_(NetName=f"/home/hlib/sem7/AI/{folder}/NN_L5_Best.hdf5",
#                 SourceDir=f"/home/hlib/sem7/AI/NewData/{dir}/",
#                 TargetFile=f"/home/hlib/sem7/AI/{d}/NN_L5_rez_{dir}",
#                 window_size=25)

        
        
        
        
        
        
#     TestNN_Custom_2(NetName=f"/home/hlib/sem7/AI/{folder}/NN_L5_Best.hdf5",
#         SourceDir=f"/home/hlib/sem7/AI/NewData/",
#         TargetFolder=f"/home/hlib/sem7/AI/{folder}/",
#         window_size=25)
    
    
    
    
    
#     for dir in ('Train', 'Test', 'Valid'):
#         TestNN_(NetName=f"/home/hlib/sem7/AI/{folder}/NN_L5_Best.hdf5",
#                 SourceDir=f"/home/hlib/sem7/AI/NewData/{dir}/",
#                 TargetFile=f"/home/hlib/sem7/AI/{d}/NN_L5_rez_{dir}",
#                 window_size=25)
    
    # print('---end  Learn---', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
#     print(f"start test for {neuro} neurons")
#     try: 
#     TestNN_Custom_2(NetName=f"/home/hlib/sem7/AI/{folder}/NN_L5_Best.hdf5",
#         SourceDir=f"/home/hlib/sem7/AI/NewData/",
#         TargetFolder=f"/home/hlib/sem7/AI/{d}/",
#         window_size=25)
#     except:
#         print("got error")







def Learn_NN_5L_Custom_2(TrainDir,ValidDir, RezDir,NN_Name, Neurons=40, Epochs=30, window_size=25, windoe_fuction='hann', optimizerFunc=RMSprop()):
    Source_Samples, Input_Files= Load_Wav_(TrainDir)
    print('end load train data', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    Source_Samples, Input_Files = RandomozeArrays(Source_Samples, Input_Files)
    print('end Randomize train data', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    
    
    X_Train, Y_Train = Convert_Wav_To_specgram(SamplesList=Source_Samples, Input_Files=Input_Files,
                                           window_size=window_size, windoe_fuction=windoe_fuction)
    print('end convert train data', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
     
    Source_Samples, Input_Files = Load_Wav_(ValidDir)
    print('end load valid data', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    X_val1, y_val1 = Convert_Wav_To_specgram(SamplesList=Source_Samples, Input_Files=Input_Files,
                                                     window_size=window_size, windoe_fuction=windoe_fuction)
    print('end convert valid data', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    input_shape = (X_Train.shape[1], X_Train.shape[2], 1)
    model = Sequential()

    model.add(BatchNormalization(input_shape = input_shape))
    model.add(Convolution2D(48, (5, 5), strides = (3, 3), padding = 'same',input_shape = input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding = 'same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(40, (3, 3), strides = (2, 2), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Flatten())
#     model.add(Dense(45))
#     model.add(Activation('relu'))
#     model.add(Dense(15))
#     model.add(Activation('relu'))
#     model.add(Dense(10))
#     model.add(Activation('relu'))
    model.add(Dense(Neurons))
    model.add(Activation('relu'))
    model.add(Dense(3))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=optimizerFunc, metrics=['accuracy'])
    csv_logger = CSVLogger(RezDir+NN_Name+'_training__log.csv', separator=',', append=False)

    
    checkpoint = ModelCheckpoint(filepath=RezDir+NN_Name+'_Best.hdf5',
                 monitor='val_accuracy',
                 save_best_only=True,
                 mode='max',
                 verbose=1)
    model.fit(X_Train, Y_Train,
          batch_size = 64,
          epochs = Epochs,shuffle=True,
          validation_data=(X_val1, y_val1),
          callbacks=[checkpoint, csv_logger])
    model.save(filepath=RezDir+NN_Name+'_Final.hdf5')
    
    
    
    
    
    
    
    
# d = 'Lab5_experiment'
# stopped at Nadam 105
# for neuro in [15]: 
# for neuro in range (5, 201, 5):
# for optimizer, optimizerFunc in zip(['Nadam'], [Nadam]):
# for optimizer, optimizerFunc in zip(['Ftrl'], [Ftrl]):
# #for optimizer, optimizerFunc in zip(['Adam', 'Adagrad', 'SGD', 'RMSprop', 'Adadelta', 'Adamax', 'Nadam', 'Ftrl'], [Adam, Adagrad, SGD, RMSprop, Adadelta, Adamax, Nadam, Ftrl]):
# #     for neuro in range (130, 400, 25): 
#     for neuro in range (30, 400, 25):
#         folder=f"Results{neuro}Neurons_{optimizer}"
#         folder = d + "/" + folder

#         e = neuro // 15

#         print(f"start train for {neuro} neurons, {optimizer}, optimizer")

#         Learn_NN_5L_Custom_2(TrainDir='/home/hlib/sem7/AI/NewData/Train/', 
#                      ValidDir='/home/hlib/sem7/AI/NewData/Valid/', 
#                      RezDir=f"/home/hlib/sem7/AI/{folder}/", 
#                      NN_Name='NN_L5', Neurons=neuro, Epochs=20+e, window_size=25, windoe_fuction='hann',optimizerFunc=optimizerFunc())


#         NetName = f"/home/hlib/sem7/AI/{folder}/NN_L5_Best.hdf5"
#         TargetFile = f"/home/hlib/sem7/AI/{d}/"

#         f = open(TargetFile+'Report.csv', 'a', newline='\n')
#         f.write('NetName = %s,'%(NetName))
#         f.write(f"{neuro},{optimizer},")
#         f.close()

#         for dir in ('Train', 'Test', 'Valid'):
#             TestNN_Custom_2(NetName=NetName,
#                     SourceDir=f"/home/hlib/sem7/AI/NewData/{dir}/",
#                     TargetFile=TargetFile,
#                     window_size=25)

#         f = open(TargetFile+'Report.csv', 'a', newline='\n')
#         f.write('\n')
#         f.close()   

#         print(f"finish train for {neuro} neurons, {optimizer}, optimizer")




folder = "Lab6_experiment"

root = r"C:/Users/hteren/unic/AI"

Learn_NN_5L_Custom_2(TrainDir=f"{root}/NewData/Train/", 
                ValidDir=f"{root}/NewData/Valid/", 
                RezDir=f"{root}/{folder}/", 
                NN_Name='NN_L5', Neurons=205, Epochs=30, window_size=25, windoe_fuction='hann',optimizerFunc=RMSprop())


# folder = 'Lab4_experiment' + "/" + folder

# print('---start Learn---', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
# Learn_NN_5L_(TrainDir='/home/hlib/sem7/AI/NewData/Train/', 
#              ValidDir='/home/hlib/sem7/AI/NewData/Valid/', 
#              RezDir=f"/home/hlib/sem7/AI/{folder}/", 
#              NN_Name='NN_L5', Epochs=25, window_size=25, windoe_fuction='hann')

# print('---end  Learn---', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

for dir in ('Train', 'Test', 'Valid'):
    TestNN_(NetName=f"{root}/{folder}/NN_L5_Best.hdf5",
            SourceDir=f"{root}/NewData/{dir}/",
            TargetFile=f"{root}/{folder}/NN_L5_rez_{dir}",
            window_size=25)