# Transformer-electricity_demand_forecasting
This project is based on the Transformer model conducted in November 2023

1. Introduction

전력 수요 예측은 전력계통의 효율적인 운영과 전력시장의 합리적인 가격 결정에 있어 중요하다. 또한 전력은 저장이 불가능하기 때문에 수요 예측을 바탕으로 미리 준비해야 하기 때문에 매우 중요하다. 딥러닝을 이용한 전력수요 예측은 실험적인 수준이 많으며 아직 실제로 적용된 사례는 많지 않다. 따라서 이번 연구에서는 Transformer를 이용한 다음날 전력 수요 예측을 하려고 한다. 데이터셋은 DACON의 한국에너지공단에서 주최한 전력사용량 예측 AI 경진대회 데이터셋을 사용했다. 데이터셋은 Z-score로 전처리 했으며 전처리한 데이터를 기준으로 Loss는 MSE(Mean Square Error)=0.0695, MAE(Mean absolute error)=0.1863이 나왔다.

 




2. Methods
   
2.1 Data Information

데이터셋은 DACON의 한국에너지공단에서 주최한 전력사용량 예측 AI 경진대회 데이터셋을 사용했다. 데이터의 대한 정보는 아래와 같다

<img width="864" alt="image" src="https://github.com/user-attachments/assets/751c83e5-7562-4df8-b33b-43917dc57771">

 
데이터 Feature(Target 포함)에 대한 정보는 아래와 같다

<img width="864" alt="image" src="https://github.com/user-attachments/assets/d752d5e8-8e49-4fc5-8db2-e638b54174af">
	 


 
2.2 Data Preprocessing



2.2.1 강수량(mm), 일조(hr) 열 제거

Feature 중에 강수량과 일조 2개의 열은 0인 값들이 매우 많아서 학습에 큰 의미가 없을 것 같아서 제거했다


 
2.2.2 데이터 간격을 1시간에서 1일로 변경

데이터셋은 1시간 간격으로 있다. Input 데이터를 넣어서 다음 Step에 대한 예측을 한다면 1시간 뒤를 예측하게 되는 것이다. 하지만 이번 연구는 다음날에 대한 전력을 예측하는 것이기 때문에 하루를 24개로 나타내는 데이터를 평균을 구해 하루를 1개로 나타내는 값으로 바꿔줬다.

<img width="864" alt="image" src="https://github.com/user-attachments/assets/9edd491c-de5e-4f25-bf26-7d31f6315d19">



2.2.3 날짜와 관련된 열 추가

전력은 날짜와 상관이 있다. 특히 주말보다 평일의 전력수요가 크기 때문에 날짜와 관련된 Feature를 추가한다.

<img width="866" alt="image" src="https://github.com/user-attachments/assets/6cb8c0fa-5e7d-4a35-945e-63c1b3b1b048">



2.2.4 Z-score로 변환

Feature끼리 스케일 차이가 크기 때문에 Z-score로 변환했다.

 

2.2.5 Prepare sequential training feature & label, Data split

Input Data Sequence Length는 7(7일)로 했고 Output은 다음날의 전력이기 때문에 Input의 두 번째 인덱스부터 마지막 인덱스에 해당하는 다음 step에 있는 값을 Target으로 사용한다. 그러기 위해서 Sliding Window(window size = 1)를 사용하여 데이터를 준비한다.

<img width="673" alt="image" src="https://github.com/user-attachments/assets/bf8e039a-879f-4514-b87b-663a3b1905fc">


 
준비된 데이터 개수는 아래와 같다

<img width="867" alt="image" src="https://github.com/user-attachments/assets/e772cb4a-3147-4adc-87b0-a98df873562b">
 



2.3 Model - Transformer

이전까지의 sequential modeling은 state transition에 기반을 뒀다. 하지만 state의 거리가 멀어질수록 정보를 반영하기가 어렵다. 이러한 경우 병렬 연산이 필요한데 Transformer가 그러기 때문에 Model로 선택했다. Transformer는 자연어처리에서 사용되던 모델이다. 따라서 입력과 출력의 길이가 다르기 때문에 Encoder-Decoder 구조로 나눠져 있다. 하지만 이번 연구에서는 입력과 출력의 길이가 똑같아도 되기 때문에 Encoder 부분만 사용했다. Transformer는 RNN을 사용하지 않고 Multi-head Attention을 사용한다. 이는 Attention을 자기 자신에게 취하는 Self-Attention을 병렬로 계산하여 관심 있는 item을 다른 item들과 비교하여 현재의 context 내에서 서로의 연관성을 계산한다. 이렇게 되면 좀 더 다양한 관계를 알 수 있다.

 

어텐션(Attention) 기법

어텐션 기법은 디코더의 매 출력 시점마다, 인코더의 출력값을 탐생하여 입력 시퀀스에 예측 결과에 더 큰 영향을 미치는 부분을 찾을 수 있는 기법이다. 어텐션 연산은 시퀀스 데이터가 주어졌을 때, 어텐션 함수를 통해 연산을 수행한다. 입력 데이터를 고정된 길이의 문맥 벡터에 인코딩할 필요가 없으며 입력 데이터에서 타겟 데이터를 생성하는데 필요한 부분에 집중하므로 신경망 모델의 장기 의존성 문제와 계산의 병렬화 문제를 개선할 수 있다는 장점이 있다. 어텐션 기법의 종류에는 하나의 시퀀스 데이터에 대하여 어텐션 연산을 수행하는 셀프-어텐션(self-attention)과 모델의 인코더 출력과 현 시점의 디코더 입력에 대해 어텐션 연산을 수행하는 인코더-디코더 어텐션이 있다.


 
![image](https://github.com/twyoon99/Transformer-electricity_demand_forecasting/assets/118956433/a54b39ac-38f0-45bb-a143-53f68f087635)

트랜스포머(Transformer) 모델 구조
 
 


2.4 Train

하이퍼 파라미터는 다음과 같다

<img width="863" alt="image" src="https://github.com/user-attachments/assets/dc470bfb-1bfb-47cc-a529-31e49b7ff978">



추가로 매 에포크마다 모델과 Loss 값을 저장시켰으며 추가적으로 MAE(Mean absolute error)도 확인지표로 사용했다.




 
3. Result

   
3.1 Loss Curve
   
<img width="603" alt="image" src="https://github.com/user-attachments/assets/6e234308-8cc2-4698-8293-ed564bfdf6aa">
 

3.2 Test 데이터셋 시각화(파랑=예측, 빨강=정답)

<img width="861" alt="image" src="https://github.com/user-attachments/assets/0d1c4d5a-9e90-4936-9952-05d68c61ef26">

400개 Test 데이터셋 중에 100개의 데이터셋을 시각화해 봤다. y값이 들쭉날쭉한 이유는 Test 데이터셋이 랜덤으로 섞여있기 때문이다. 시각화는 Output의 마지막 값만 사용했다. 왜냐면 그 값이 우리가 알고자 하는 다음날의 전력이기 때문이다.

 

3.3 Test Loss

<img width="865" alt="image" src="https://github.com/user-attachments/assets/faca861a-65f0-43e8-a40f-666a0ac989f5">

 


4. Discussion

이번 연구는 Transformer를 이용해 Single-step에 대한 전력 수요 예측을 했다. 단일 step을 예측하기 때문에 Transformer의 블록을 1개만 사용해도 Test Dataset에 대해서 시각화와 평가지표를 확인했을 때도 충분히 좋은 성능을 나타냄을 알 수 있다. 이번 연구로 완성된 모델은 다음날 전력 예측을 위한 모델로 사용될 수 있을 것이다. 원래는 Single-step이 아닌 여러 개의 스텝을 예측하는 Multi-step 예측을 하려고 했다. 여러 개의 미래를 예측한다는 것은 상식적으로 생각해도 쉽지 않고 실제로도 그렇다. 즉, 기본적인 Transformer 구조보다 더 복잡하고 응용된 Model이 필요하다. 레퍼런스를 찾다가 알게 된 Model 중에 하나로 Temporal Fusion Transformer가 있다. 구조가 복잡하여 이번 연구에 사용된 Transformer와 다른 점 몇 가지만 얘기하면 Encoder-Decoder 구조를 사용하며 LSTM도 같이 사용한다. 그리고 미래 시점의 값을 현재 시점에서도 알 수 있는 변수(Time Varying Known Input)를 입력에 같이 넣는다. 위에서 이번 연구에서 Single-step을 하기 전에 Multi-step 예측을 먼저 시도했다고 언급했는데 Multi-step 모델 구조 설계를 생각하면서 날짜와 같은 변수(Time Varying Known Input)를 입력에 같이 넣어주면 어떨까라는 생각을 했는데 실제로 Temporal Fusion Transformer에서도 그러한 개념이 사용됐다는 게 흥미로웠다. Multi-step 예측 모델을 만들지 못한 이유는 아직 Transformer에 대해서 제대로 이해가 안 됐고 따라서 모델 설계도 쉽지 않았기 때문이다. 나중에 Transformer에 대해서 더 잘 알게 된다면 Multi-step 예측 모델도 만들어보고 싶다.
