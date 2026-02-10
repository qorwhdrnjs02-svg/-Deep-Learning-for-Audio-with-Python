import os #os에 직접 접근 -> window 운영 체제의 파일을 직접 다룰 수 있음(walk 함수 이용)
import librosa
import math
import json

DATASET_PATH = "music"
JSON_PATH = "data.json"

SAMPLE_RATE  = 22050
DURATION = 30 #measured in second
SAMPLE_PER_TRACK = SAMPLE_RATE * DURATION

#mfcc를 저장하는 함수
#dataset_path는 파일의 경로
# json_path를 통해 파일 변환
    # JSON은 wav파일을 직접 다루기엔(sample rate * 노래길이 만큼의 데이터 개수)
    # 파일이 너무 크니 컴퓨터와 사람이 함께 공유할 수 있는 데이터 형식을 의미
        #JSON의 가변성
        #JSON은 고정된 형식이 아니라, 우리가 정의한 데이터 구조이기 때문에 
        #기준을 어떻게 잡느냐애 따라 내용이 달라짐
        #MFCC =13은 소리의 특징을 가장 잘 나타내는 핵심 요약본으로 가장 적절하기 때문
        #데이터 크기를 획기적으로 줄여서 컴퓨터의 연산 부담을 덜어내고 학습 정확도를 높임
        #만약 개별 파일을 FFT하여 frequency로 파일 정보를 저장하고 다루려 한다면
        #휠씬 방대하고 복잡한 구조의 JSON 파일이 만들어짐
#n_mfcc = 13는 wav파일의 특성구분을 13개로 하겠다는 의미
#n_fft = 2048 은 정해둔 sr 중에서(지정해주지 않으면 lirosa는 22050으로 설정) 
        #2048개만큼 분석해서 해당 구간을 이루는 주파수를 
        #나누어 보겠다는 의미
# hop_length = 512 는 mfcc 상에서 연속적인 분석을 위해 sample을 겹치는 정도
# num_segment = 5는 파일의 재생시간을(우리의 경우 30초)5등분 하겠다는 의미(여기서는 6초씩)
# 종합하면 dataset_path에 있는 파일을
# JSON 파일로 저장하는데
# 13개의 특성으로 분리할 거고
# 한개의 파일을 5등분해서 학습 데이터를 뻥튀기 한다음에
# 6초 가량의 뻥튀기된 파일의 1초를 sr 로 나누어 
# 그 중 또 2048만큼만 나눠서 볼건데 
# 연속적인 분석결과가 필요하기 때문에 1~2048 -> 513~2560 이렇게 분석을 연속화 할거임 
# (한 노래(30초)당 5*6*22050개의 샘플을 2048씩 가져와서 분석, 단 약 75%(hop_length)씩 겹쳐서)
# 조금더 정리된 언어로는 
# "6초(132,300개 샘플)마다 별도의 MFCC 행렬을 만들어서, 
#  총 5개의 데이터 꾸러미를 JSON에 담는다."
def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048, 
              hop_length=512, num_segments=5):
    
    #dictionary to store data
    data = {
        "mapping": [],
        "mfcc" : [],
        "labels": []
    }
        # # ---------------------------------------------------------------------------
        # # [ data 딕셔너리 저장 상태 시뮬레이션: hiphop.00004.wav 처리 후 ]
        # # ---------------------------------------------------------------------------
        # data = {
        #     # 1. mapping: 장르의 '이름표' (0번: classical, 1번: blues, 2번: hiphop)
        #     "mapping": ["classical", "blues", "hiphop"],

        #     # 2. mfcc: 실제 모델이 학습할 '숫자 지도' (데이터 본체)
        #     # 한 곡을 5등분(num_segments=5)했으므로, 1곡당 5개의 행렬이 추가됨
        #     "mfcc": [
        #         [[...], [...], ...], # hiphop.00004.wav의 1번째 6초 구간 (MFCC 행렬)
        #         [[...], [...], ...], # hiphop.00004.wav의 2번째 6초 구간 (MFCC 행렬)
        #         [[...], [...], ...], # hiphop.00004.wav의 3번째 6초 구간 (MFCC 행렬)
        #         [[...], [...], ...], # hiphop.00004.wav의 4번째 6초 구간 (MFCC 행렬)
        #         [[...], [...], ...]  # hiphop.00004.wav의 5번째 6초 구간 (MFCC 행렬)
        #     ],

        #     # 3. labels: 각 MFCC 덩어리가 어떤 장르인지 알려주는 '정답 번호'
        #     # hiphop은 mapping 리스트의 2번 인덱스이므로 숫자 '2'를 5번 기록함
        #     "labels": [2, 2, 2, 2, 2] 
        # }
        # # ---------------------------------------------------------------------------
        # # TIP: mfcc 행렬의 첫 번째 리스트 덩어리 (6초 구간 하나)
        #   [
        #     [-12.4, 3.5, ..., 0.1], # t=0.02초 지점의 MFCC 계수 13개
        #     [-11.8, 3.2, ..., 0.2], # t=0.04초 지점의 MFCC 계수 13개
        #     ... 
        #     [-15.1, 2.8, ..., -0.1] # t=6.00초 지점의 MFCC 계수 13개 
        #     (약 6초 * 22050(sr) / 512(hop_length) = 258번째 줄)
        #   ]
        # ] 
        # 즉 mfcc는 3차원 배열로서 총 5개의 리스트를 저장하는데
        # 하나의 리스트는 약 258개의 리스트를 저장하고
        # 258개의 리스트는 각각 13개의 실수를 저장
        # # ---------------------------------------------------------------------------
    
    # loop through all the genres
    # walk 함수는 dataset_path에 넣어준 경로를 시작으로
    # 가장 하위 폴더까지 파고들도록 도와주는 함수임
    # 따라서 Data/genres/를 dataset_path로 입력하면
    # Data/genres/blues, Data/genres/hiphop등으로 루프를 도는데
    # filenames는 Data/genres/blues 안에 있는 파일들을 의미하는거임
      #주의할 점은 dataset_path 하위 폴더에 Data/genres/music/hiphop 등으로
      #하위 폴더를 하나 더 두게 되면 아래 if 문이 불필요하게 music이라는 장르를
      #blues나 힙합같은 세부장르와 같은 범위로 인식할 수 가 있으므로 주의가 필요함

    #말 그대로 부분당 샘플의 개수로서, SAMPLE_PER_TRACK은 트책당 총 샘플의 개수고
    # 그걸 num_segments = 5로 나누었으니, 6초의 segment당 몇개의 샘플이 있는지에 대한 정의
    num_samples_per_segment = int(SAMPLE_PER_TRACK / num_segments)
    
    # 딥러닝 모델은 입력 데이터 즉 우리가 학습시키는 데이터의 크기가 모두 동일해야 하는데
    # 우리가 학습을 위해준비한 모델이 30초를 조금 모자랄 때에 대비한 안전장치이다.
    # 앞서 적었듯이 우리는 하나의 segment당
    #(약 6초 * 22050(sr) / 512(hop_length) = 258)개의 가로 줄이 일관되기 그려질 수 있도록
    # 강제로 ceil(올림)하는 것.
    # 이렇게 함으로써 하나의 segment에 대해 mfcc는 13개의 세로줄, 258개의 가로줄이 생김
      #이때, 그어야 하는 줄이 258개를 넘어가면 어떻게 하냐라고 생각할 수 있는데
      #이는 finish_sample을 통해 잘라놨기 때문에 고려하지 않아도 됨
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length) #1.2 -> 2

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        #ensure that we're not at the root level
        #walk함수가 시작하는 폴더의 경로와 하위 폴더의 이름이 다르면
        if dirpath is not dataset_path:

            #save the semantic label
            #파일 경로를 나누어 
            dirpath_components = dirpath.split("/") 
            #genre/blues => []"genre", "blues"] 장르이름만 추출하고
            #그중 가장 마지막 컴포넌트만 label 을 붙여서
            sementic_label = os.path.split(dirpath)[-1]
            # mapping하며,
            # mapping 리스트에 추가된 순서(인덱스)가 
            # 곧 해당 장르의 숫자 ID가 됨 (예: 0=blues, 1=classical)
            data["mapping"].append(sementic_label)
            print("\nProcessing {}".format(sementic_label))




            # process file for a specific genre

            #filenames라는 리스트에는 각각의 음원 파일이 저장되어 있을거고
            # 그걸 f에 저장하니까 f에는 파일 이름이 저장되고
            for f in filenames :

                #load audio file
                #f에는 파일 이름이있고,
                #os.path.join(dirpath, f)를 쓰면
                #앞서 정의한 dirpath = Data\genres\blues 에 자동으로 역슬래시를 붙여
                # Data/genres/blues/blues.00001.wav 로 파일 경로를 저장함
                file_path = os.path.join(dirpath, f)

                try:
                #이제 경로를 통해 파일을 불러오고

                    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                except Exception as e:
                    print("f\nSkipping {file_path}: {e}")
                    continue

                #process segment ectracting mfcc and storing data
                #우리가 num_segment를 정의했던 이유가 하나의 음원 파일을 5조각으로 나누어
                #학습 데이터를 뻥튀기 시키기 위함 이었는데, 아래 for 문이 이를 본격화하는 알고리즘임
                # 당연히 for문은 0~5까지 루프를 돌면서
                for s in range(num_segments):
                    #s=0일때 start는 0, finish는 segment 하나의 마지막 샘플까지 가 되고
                    # 이를 s<5일 때까지 반복하니까 총 5개의 start와 finish가 서로의 끝과 끝을
                    # 이어서 하나의 음원을 이루게 됨
                    start_sample = num_samples_per_segment * s # s=0 -> 0
                    finish_sample = start_sample + num_samples_per_segment # s=0 -> num_samples_per_segment 
                    

                    #그렇게 나누어진 start와 finish를 signal에 대해 for문을 통해 나누고 mfcc로 저장함
                    # 즉 하나의 음원에 대해 5개의 mfcc가 생기는 것
                    mfcc = librosa.feature.mfcc(y=signal[start_sample:finish_sample],
                                                sr=sr,
                                                n_mfcc=n_mfcc, n_fft=n_fft,
                                                hop_length=hop_length)
        

                    # store mfcc for segment if it has the expected length

                    #우리가 올림하여 258이 되도록한 mfcc_vector_per_segment에 대해
                    #만약 실제로 하나의 mfcc가 가지고 있는 array의 길이가 258이 맞다면
                    #(단, 그냥 mfcc의 길이로 하면 13이 되므로 transpose 해서)
                    mfcc = mfcc.T
                    if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                        #data라는 dictionary 안 mfcc key를 가진 리스트에
                        #mfcc를 list화 시켜서 저장(Numpy 배열은 append 불가)
                        data["mfcc"].append(mfcc.tolist())


                        # 여기서 i는 가장 처음 for문의 i
                        # 1을 빼는 이뉴는 i=0은 walk가 시작하는 바로 그 폴더이고
                        # 거기서 split된 장르의 이름은 genre이기 때문에
                        # 우리가 목표하는 바가 아님 blues 등 세부 장르가 
                        # labels 배열의 0번째 원소가 되도록 하기 위함
                        data["labels"].append(i-1)

                        
                        #해당 if문을 거치만 예를 들어 2번 폴더가 힙합일때
                        #hiphop 안에 있는 파일들의 segment는 모두 i=2일 라는
                        #label이 붙게 되는 것
                        #따라서 
                    #data = {
                    #     "mapping": [],
                    #     "mfcc" : [],
                    #     "labels": []
                    #        }
                    #해당 dictionary 안에서 
                    #hiphop이라고 mapping할 수 있고
                    #hiphop의 륵징은
                    #mfcc 안에 저장된 258개의 13개의 원소를 가진 
                    #list를 통해 나타나며
                    #labels = 2이라고 분류하겠습니다
                        #로 정리되는 것임


                        #여기서 s는 3번째 for문
                        print("{}, segment: {}".format(file_path, s+1))
   
    #우리가 저장한 data dictionary 파일을 json 파일로(indent=4로 들여쓰기하여) 저장
    with open (json_path, "w") as fp:
        json.dump(data, fp, indent=4)
    
if __name__ == "__main__":
    save_mfcc(DATASET_PATH,JSON_PATH,num_segments=10)
