import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np

file = "blues.00000.wav"

#waveform
# 우리가 지난번에 배웠듯이 sr (단위가 Hz)만큼 초당 n가의 점을 시간 축 위에 찍으면
# 해당 점에 대응하는 아날로그 신호를 quantize 해서 대략적인 값을 저장하는것이
# ADC(analog digital conversion)이다
# signal은 각 sample에 대응하는 quantize된 값들을 저장하는 배열이고
# sr은 말 그대로 sample rate다
# 즉 우리가 다루는 음원이 30초 짜리니까 sr=22050이면 총 22050*30개의 samle이 생기고
# 배열인 signal 역시 각 샘플에 대응하는 값들이 저장되므로
# 22050 * 30개의 값이 저장된다
signal, sr = librosa.load(file, sr=22050) #sr * T -> 22050 * 30
    #왜 하필 22050인가 -> 일반적으로 인간이 들을 수 있는 가청 주파수가 20000hz이고
    #나이퀴스트 정리에 의하면 어떤 주파수를 제대로 기록하려면, 그 주파수보다 최소
    #2배 더 빠르게 점을 찍어야 한다
    # 그게 일반적인 디지털 음원이 44100hz를 사용하는 이유인데
    # 계산량을 줄이기 위해 딥러닝에서 일반적으로 22050hz를 sr로서 자주 이용한다
# print(f"signal 타입: {type(signal)}")
# print(f"signal의 처음 5개 값: {signal[:5]}") # -1~1 사이의 소수들
# print(f"데이터 전체 개수: {len(signal)}")
# print("-" * 30)
# print(f"sr 타입: {type(sr)}")
# print(f"설정된 SR 값: {sr}")
    # signal과 sr에는 각각 다음과 같은 값들이 저장된다
    # signal 타입: <class 'numpy.ndarray'>
    # signal의 처음 5개 값: [ 0.00732422  0.01660156  0.00762939 -0.00350952 -0.0022583 ]
    # 데이터 전체 개수: 661794 -> 소스점 단위까지 계산하므로 정확히 나누어 떨이지는 않음
    # ------------------------------
    # sr 타입: <class 'int'>
    # 설정된 SR 값: 22050
# librosa.display.waveshow(signal, sr=sr)
# plt.xlabel("Time")
# plt.ylabel("Aplitude")
# plt.show()

# fft -> spectrum
# [FFT: Fast Fourier Transform]
# 1. DFT의 원리: 연속적인 아날로그 신호를 sr(샘플링 레이트) 주기로 채취하여
#    (원래 analog 함수와 impulse train의 곱) 
#    이산 신호(Discrete Signal)로 만들고, 이를 푸리에 변환하는 과정.
# 2. FFT의 효율성: 데이터 개수가 N일 때, 일반 DFT의 연산량 O(N^2)을 
#    O(N log N)으로 획기적으로 줄여 실시간 분석을 가능하게 함.
# 3. 데이터 구조: 입력인 signal 배열이 N개의 샘플(sr * T)을 가지므로, 
#    출력인 fft 배열 역시 동일하게 N개의 복소수 원소를 가짐.
fft = np.fft.fft(signal)

# 앞서 배웠듯이 magnitude는 각 주파수가 얼만큼 전체 소리에 기여하는지 나타냄
# FT의 결과가 복소수(크기와 방향을 가짐)인데, 
# 우리가 필요한 것은 주파수에 대응하는 에너지의 크기이니
# 절대값(absolute) 값이 필요하고
# (역시 magnitude는 fft라는 배열의 각 요소의 절대값만 계산하였으므로
# fft와 데이터 개수가 같음)
magnitude = np.abs(fft)

# FT를 통해 주파수 domain에 해당하는 y값인 magnitude를 얻었지만
# 아직 각 주파수에 대해 magnitude를 대응 시키지 않았음
# 해당 함수가 그 과정인데, domain의 범위가 0부터 sr만큼인 이유에 대해서
# 헷갈릴 수 있지만 주파수와 시간은 다른개념(정확히는 역의 개념) 임을 상기하자.
# 우리는 애초에 sr(sample rate)을 0부터 22050으로 잘랐고
# 그 말인 즉슨 0부터 11025Hz(by 나이퀴스트 정리)의 소리까지만 관심 있다는 뜻 
# (주파수가 높을수록 높은 소리라는 것을 상기.)
# 즉 우리가 관심있는 소리의 높이 영역에 있어서 그것들이 얼만큼 음원에서 비율을 차지하는지는
# 0부터 sr영역에서 모두 알 수 있음
frequency = np.linspace(0, sr, len(magnitude))

# 절반으로 자르는 이유는 어차피 주기 함수이기 때문에 
# 1. 이유: 실수 신호를 FFT하면 결과가 sr/2를 기점으로 대칭(Symmetry)을 이룸.
# 2. 원리: sin/cos 함수처럼 정보가 중복되므로, 실제 의미 있는 양의 주파수 영역만 사용.
# 3. 범위: 0부터 나이퀴스트 주파수(sr/2)까지가 우리가 물리적으로 신뢰할 수 있는 소리 정보임.
left_frequency = frequency[:int(len(frequency)/2)]
left_magnitude = magnitude[:int(len(frequency)/2)]

# plt.plot(left_frequency, left_magnitude)
# plt.xlabel("Frequency")
# plt.ylabel("Magnitude")
# plt.show()

# stft -> spectogram

#stft의 의미는 short_time 즉 1초를 더~ 짧게 들여다보고 분석하겠다는 의미
#따라서 n_fft = 2048로 설정한다는 것은 우리가 sr로 정한 22050개의 샘플(1~22050) 중에서
#가장 앞의 1~2048 개의 샘플을 가져와서 fft 하여 분석하겠다는 의미 
n_fft = 2048
#hop_length를 설정하는 이유는 스펙토그램 상에서 그려지는 
#연속적인 시간당 주파수의 분포가
#부드럽게 이어지기 위해서는 겹쳐가면서(데셍할 때 음영을 칠하듯) 그려야 하고
#그 정도를 정하는 것.
# 즉 hop_length가 512 라는 것은 1~2048번쨰 샘플을 그래프 위에 표현하고
# 그 다은에는 513~2560번째 샘플을 그래프 위에 표현하고 ... ~66만개까지 반복되는 것.
hop_length = 512

# 이렇게 하면 stft는 시간-주파수 평면 위에 소리의 크기(amlitude) 를 그리게 되는데
# 시간축은 signal 배열(전체 22050*30 개의 배열)을 hop_length(512)만큼 나눠
# 약 1292개의 칸(한칸당 약 0.023초) 으로 표현함

# n_fft는 주파수 축을 결정하는데, 결국엔 stft 역시 fft를 
# 아주 짧은 시간에 대해 연산, 적분하는 것이고
# 그에 따라 주파수 domain이 정해짐
# (즉 n_fft는 시간을 얼마나 짤라서 가져오냐를 결정하기도 하지만
# freqency를 그리는 데에도 중요한 역할을 한다는 뜻)
# 주파수 축은 정확히 n_fft/2 + 1 의 칸이 나옴
stft = librosa.core.stft(signal, hop_length=hop_length, n_fft=n_fft)

#조금 더 정리된 언어로는
# [STFT 파라미터 정의 및 본질 이해]

# 1. n_fft (Window Size): 
# - 분석할 '시간의 덩어리'이자 '주파수 해상도'를 결정하는 핵심 변수.
# - 1초(sr)를 더 짧게 쪼개어 약 0.09초(2048개 샘플)를 한 단위로 FFT를 수행함.
# - 원리: 적분 구간(시간)을 어떻게 설정하느냐가 출력될 주파수(Frequency)의 정밀도에 직접 영향을 미침.
# - Trade-off: n_fft를 키우면 주파수 분석은 정밀해지나 시간 정보가 뭉개지고, 줄이면 시간은 정확해지나 주파수 구분이 흐려짐.

# 2. hop_length (Step Size):
# - 분석 창문(n_fft)을 옆으로 미는 보폭. 지운 님의 비유처럼 '데생의 음영'을 넣듯 겹쳐서 스캔함.
# - n_fft(2048)보다 작은 512를 설정하여 샘플 간의 연속성을 확보하고 부드러운 시간당 주파수 분포를 그려냄.
# - 가로축(Time) 한 칸의 실제 시간 단위는 'hop_length / sr' 초가 됨.

# 3. STFT 실행:
# - 짧게 자른 시간 덩어리(n_fft)마다 FFT를 반복 수행하여 '시간-주파수-데시벨'의 3차원 지도를 생성함.



# 역시나 해당 stft 배열의 값은 허수이기 때문에 절대값을 취하고
spectrogram = np.abs(stft)
# 절대값에 대해 log를 취해서 amplitude를 우리가 일반적으로 사용하는 dB로 변환
log_spectogram = librosa.amplitude_to_db(spectrogram)
# 스펙토그램을 그림
# librosa.display.specshow(log_spectogram, sr=sr, hop_length=hop_length)
# plt.xlabel("Time")
# plt.ylabel("Frequency")
# plt.colorbar()
# plt.show()

# MFCCs

#여기서 'y'는 y축을 의미하는 것이 아니라 신호처리에서 관습적으로 사용하는 종속변수로서
#신호 데이터 자체 를 뜻함

#기본적으로 spectogram을 그리는 방식과 유사하지만 mfcc에 대한 이해가 필요함
# mfcc는 기본적으로 사람이 느끼기에 해당 source가 어떤 음색을 보이는지 이해하게
# 도와주는 방법임.

# 특히 n_mfcc = 13에 대하여 이해할 필요가 있는데
# 이는 음원을 13개의 '특성'으로 나누어 시간에 대하여 해당 특성이 어떤 경향을 보이는지
# 알려줄 수 있도록 하는 변수라고 행각하면됨
# 즉 해당 그래프의 세로축은 주파수대역이 아니라 주파수 대역을 일정한 알고리즘을 따라
# 분류하고, 재조합 한다음에

#----------------------------
# [MFCC 1~13번 계수의 역할: 소리의 '지문'을 그리는 13개의 붓터치]
# 각 계수는 주파수 대역이 아니라, 소리의 '형태(Envelope)'를 결정하는 수학적 성분임.

# 1번 (MFCC 0) - 전체 에너지 (DC 성분)
# : 소리의 총 합산 크기(볼륨)를 결정. 
# : [비유] 얼굴의 '전체 크기'와 밝기

# 2번 (MFCC 1) - 에너지의 기울기 (Spectral Slope)
# : 저음이 강한지 고음이 강한지 결정. 
# : [비유] 얼굴의 '상하 비율' (긴 얼굴 vs 짧은 얼굴)

# 3번 (MFCC 2) - 중심부의 굴곡
# : 특정 대역(주로 중음역)의 에너지가 솟았는지 눌렸는지 나타냄.
# : [비유] 이목구비의 '굵직한 위치'

# 4~13번 - 세밀한 음색 패턴
# : 소리의 겉모양(Envelope)이 얼마나 복잡하게 요동치는지 나타냄.
# : [비유] 눈매, 콧날의 각도, 입술 모양 등 '세부 디테일'

# ※ 참고: 14번 이후는 소리의 종류를 구분하는 데 불필요한 '노이즈'로 간주하여 보통 제외함.
#----------------------------

# 이렇게 분류하는 것임 

mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)
librosa.display.specshow(mfccs, sr=sr, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("MFCC")
plt.colorbar()
plt.show()

# 더 정리된 언어로 다시 설명하면

# [MFCC의 본질: 주파수 대역이 아닌 '소리의 성질' 추출]
# - 13개의 칸은 연속적인 주파수 구간(Hz)을 의미하는 것이 아님.
# - 각 칸은 소리의 '전체적인 형태'를 묘사하는 서로 다른 기준들임.
# - 예: 1번은 크기, 2번은 고/저 쏠림 정도, 3번은 특정 구간의 굴곡 등...
# - 이렇게 '추상화'된 데이터이기 때문에 딥러닝 모델이 음색의 차이를 더 쉽게 학습할 수 있음.

# [MFCC 13개 계수의 의미]
# - 특정 주파수 대역을 13개로 나눈 것이 아님 (그건 Mel-Spectrogram의 역할).
# - MFCC는 그 주파수 분포를 다시 수학적으로 압축하여 소리의 '전체적인 질감/모양'을 뽑아낸 것.
# - 즉, 각 칸은 "소리의 형태를 결정하는 13가지 핵심 성분"을 의미하며, 
# - 이를 통해 모델은 주파수 전체를 보지 않고도 '음색'의 특징을 빠르게 파악함.