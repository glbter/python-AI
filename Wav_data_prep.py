import numpy as np
import os
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
from scipy.io import wavfile
import numpy as np
from sklearn.preprocessing import LabelBinarizer
labelbinarizer = LabelBinarizer()
import os
from scipy import signal
from shutil import copyfile
import random






def Convert_To_06(samples):
    sample_rate=8000
    WindowSize = 40
    LeftSeek = 300
    RezFileLength = 0.6


    if len(samples) <= int(0.6*sample_rate):
        return samples

    Samples_SUM = np.arange(len(samples), dtype='float64')
    ResultS = np.arange(int(RezFileLength * sample_rate), dtype=samples.dtype)
    Samples_Float = np.arange(len(samples), dtype='float64')

    # ------ARRAY -----AverageSum--------------------------
    Average = 0.0
    for i in range(len(samples)):
        Average += samples[i]

    Average = Average / len(samples)
    for i in range(len(samples)):
        Samples_Float[i] = abs(samples[i] - Average)

    MaxSum = 0
    TotalSum = 0
    for i in range(len(Samples_SUM)):
        Samples_SUM[i] = 0
    i = WindowSize
    while i < (len(samples) - WindowSize - 1):
        CurentSum = 0
        J = -WindowSize
        while J < WindowSize:
            CurentSum = CurentSum + Samples_Float[i + J]
            J = J + 1
        Samples_SUM[i] = CurentSum / (2 * WindowSize + 1)
        if MaxSum < Samples_SUM[i]:
            MaxSum = Samples_SUM[i]

        i += 1
    Thresholt = MaxSum / 6
    for asum in Samples_SUM:
        if Thresholt < asum:
            TotalSum += asum
    CenterIndex = 0
    CurentSum = Samples_SUM[0]

    while CurentSum < TotalSum / 2:
        CenterIndex += 1
        CurentSum += Samples_SUM[CenterIndex]

    StartIndex = 0
    BitRateCount = 0
    for i in range(len(Samples_SUM)):
        if Samples_SUM[i] > Thresholt:
            BitRateCount += 1
        else:
            BitRateCount = 0
        if BitRateCount > WindowSize * 4:
            if i > WindowSize * 4 + LeftSeek:
                StartIndex = int(i - (WindowSize * 4 + LeftSeek))
            break


    if CenterIndex < int(StartIndex + (sample_rate * RezFileLength / 2)) and (sample_rate * RezFileLength / 2) < CenterIndex:
        StartIndex = CenterIndex - (sample_rate * RezFileLength / 2)
    if StartIndex + sample_rate * RezFileLength > len(samples):
        StartIndex = len(samples) - sample_rate * RezFileLength
    StartIndex = int(StartIndex)
    for i in range(int(RezFileLength * sample_rate)):
        ResultS[i] = samples[i + StartIndex]
    return  ResultS

def ConvertTo_8K(SourceDir, TargetDirectory,Prefics, ClassType ):
    #  -------Load Files----------------------------------------
    Wav_Files = []
    for d, dirs, files in os.walk(SourceDir):
        for file in files:
            if file.endswith(".wav"):
                Wav_Files.append(file)
    #  -------Convert to 8K and Raname and Save Files----------------------------------------
    for wfile in Wav_Files:
        curerentFile = SourceDir + "/" + wfile
        sample_rate, samples = wavfile.read(curerentFile)

        i = wfile.find(".wav")
        FileName = wfile[:i]
        FileName = FileName.replace("_nohash_", "")

        if sample_rate == 16000:
            lennewarray = int(len(samples) / 2)
            f = signal.resample(samples, lennewarray)
            Samples_1 = np.arange(lennewarray, dtype=samples.dtype)
            for i in range(lennewarray):
                Samples_1[i] = int(f[i])
            curerentFile = TargetDirectory+ "/"+ Prefics + FileName + "_0000_" + ClassType + ".wav"
            Samples_final = Convert_To_06(Samples_1)
            wavfile.write(curerentFile, 8000, Samples_final)

        if sample_rate == 8000:
            curerentFile = TargetDirectory + FileName + "+0000_" + ClassType + ".wav"
            Samples_final = Convert_To_06(samples)
            wavfile.write(curerentFile, sample_rate, Samples_final)
            
            
def Divide_TrainTestValid(SourceDirectory, TrainDirectory, TestDirectory, ValidDirectory,TestPercent, ValidPercent):

    Clas1_Files = []
    Clas2_Files = []
    Clas3_Files = []
    Test_Files = []
    Clas1_ID = "cl_1"
    Clas2_ID = "cl_2"
    Clas3_ID = "cl_3"

    Wav_Files = []
    for d, dirs, files in os.walk(SourceDirectory):
        for file in files:
            if file.endswith(".wav"):
                Wav_Files.append(file)
    for fwav in Wav_Files:
        if Clas1_ID in fwav:
            Clas1_Files.append(fwav)
        if Clas2_ID in fwav:
            Clas2_Files.append(fwav)
        if Clas3_ID in fwav:
            Clas3_Files.append(fwav)
  
    randTestFiles = []
    if len(Clas1_Files) + len(Clas2_Files) < len(Clas3_Files):
        for i in range(len(Clas1_Files) + len(Clas2_Files)):
            Index = random.randint(0, len(Clas3_Files) - 1)
            randTestFiles.append(Clas3_Files[Index])
            del Clas3_Files[Index]
        Clas3_Files = randTestFiles

#     TestSize = len(Clas1_Files)
#     if TestSize > len(Clas2_Files):
#         TestSize = len(Clas2_Files)
#     if TestSize > len(Clas3_Files):
#         TestSize = len(Clas3_Files)
    Size = len(Clas1_Files) + len(Clas2_Files) + len(Clas3_Files)
    ValidSize = int(Size * ValidPercent)
    TestSize = int(Size * TestPercent)
    print(Size)
    print(ValidSize)
    print(TestSize)
    print(ValidPercent)
    print(TestPercent)
    print(int(ValidSize/3))
    print(int(TestSize/3))

    # ------------------TEST----------------------------------------

    for i in range(int(TestSize/3)):
        Index = random.randint(0, len(Clas1_Files) - 1)
        Test_Files.append(Clas1_Files[Index])
        del Clas1_Files[Index]
        Index = random.randint(0, len(Clas2_Files) - 1)
        Test_Files.append(Clas2_Files[Index])
        del Clas2_Files[Index]
        Index = random.randint(0, len(Clas3_Files) - 1)
        Test_Files.append(Clas3_Files[Index])
        del Clas3_Files[Index]

    for tFile in Test_Files:
        src = SourceDirectory + "/"+ tFile
        dst = TestDirectory + "/"+ tFile
        copyfile(src, dst)
   # ------------------TEST---------------------------------------


  # ------------------Valid----------------------------------------
    Test_Files = []
    for i in range(int(ValidSize/3)):
        Index = random.randint(0, len(Clas1_Files) - 1)
        Test_Files.append(Clas1_Files[Index])
        del Clas1_Files[Index]
        Index = random.randint(0, len(Clas2_Files) - 1)
        Test_Files.append(Clas2_Files[Index])
        del Clas2_Files[Index]
        Index = random.randint(0, len(Clas3_Files) - 1)
        Test_Files.append(Clas3_Files[Index])
        del Clas3_Files[Index]

    for tFile in Test_Files:
        src = SourceDirectory + "/"+ tFile
        dst =ValidDirectory + "/"+ tFile
        copyfile(src, dst)
    # ------------------Valid---------------------------------------


    Train_Files = Clas1_Files + Clas2_Files + Clas3_Files
    for tFile in Train_Files:
        src = SourceDirectory + "/"+  tFile
        dst = TrainDirectory + "/"+  tFile
        copyfile(src, dst)
        
        
        
        
        
        
        
# ConvertTo_8K(SourceDir=r'/home/hlib/sem7/AI/data/tree', TargetDirectory=r'/home/hlib/sem7/AI/My_data', Prefics='tree',
#                  ClassType='cl_1')
# ConvertTo_8K(SourceDir=r'/home/hlib/sem7/AI/data/go', TargetDirectory=r'/home/hlib/sem7/AI/My_data', Prefics='go',
#                  ClassType='cl_2')

for dir in os.listdir(path='/home/hlib/sem7/AI/data'):
    #if dir == 'go' or dir == 'tree': 
    #    continue

    target = f"/home/hlib/sem7/AI/My_data2/{dir}"
    os.mkdir(target)
    ConvertTo_8K(SourceDir=f"/home/hlib/sem7/AI/data/{dir}", TargetDirectory=target, Prefics='other', ClassType='cl_3')







Divide_TrainTestValid(SourceDirectory=r'/home/hlib/sem7/AI/My_data3',
                          TrainDirectory=r'/home/hlib/sem7/AI/NewData/Train',
                          TestDirectory=r'/home/hlib/sem7/AI/NewData/Test', 
                          ValidDirectory=r'/home/hlib/sem7/AI/NewData/Valid', 
                          TestPercent=0.1,
                          ValidPercent=0.1)

