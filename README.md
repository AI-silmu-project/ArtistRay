# 생채기

## Installation

1. Repo 다운로드
    ```bash
    git clone --recurse-submodules https://github.com/AI-silmu-project/Kizuato_Collection.git
    ```
2. SiamMask 파라미터 다운로드
    ```bash
    cd config
    wget http://www.robots.ox.ac.uk/~qwang/SiamMask_VOT.pth
    wget http://www.robots.ox.ac.uk/~qwang/SiamMask_DAVIS.pth
    cd ..
    ```
    또는, [링크1](http://www.robots.ox.ac.uk/~qwang/SiamMask_VOT.pth) [링크2](http://www.robots.ox.ac.uk/~qwang/SiamMask_DAVIS.pth)에서 두 파라미터 모두 다운로드 후 config 폴더에 넣어주세요.

3. 의존성 설치
    ```
    pip install -r requirements.txt
    ```
    만약 실행 시 더 필요한 패키지가 있다면, 수동으로 설치해 주세요. **PyTorch는 CUDA로 설치하는 것을 강력히 권장합니다.**
    
4. pysot, pyvotkit 빌드 
    **윈도우의 경우, VS C++ Build Tools가 필요합니다!**
    ```bash
    cd SiamMask/utils/pyvotkit
    python setup.py build_ext --inplace
    cd ../../../

    cd SiamMask/utils/pysot/utils/
    python setup.py build_ext --inplace
    cd ../../../../
    ```

## Run
```bash
python pipeline.py [CameraIndex]
```
카메라 인덱스는 기본 0이며, 두 개 이상 있는 경우 수동으로 지정해주세요.

### 조작법
`q`: 프로그램 종료
`w`: 초점거리 멀리
`s`: 초점거리 가까이
`a`: 어둡게
`d`: 밝게
`c`: 캡쳐/캡쳐 취소
`r`: 타투 재생성

1. `w,a,s,d`를 이용해 흉터가 잘 보이도록 조절한 후, `c`를 눌러 캡쳐해 주세요. 캡쳐가 마음에 들지 않으면 `c`를 다시 눌러 돌아갈 수 있습니다.
2. 캡쳐한 화면에서 흉터를 드래그하여 선택해 주세요. 선택이 끝나면 스페이스바나 엔터를 눌러주세요.
3. 조금 기다리면 흉터를 덮는 타투가 완성되며, 화면에서 타투 결과를 미리 볼 수 있습니다. 결과는 모두 `outputs` 폴더에 저장되며, 원할 경우 `outputs/output_roi.png`를 편집하여 타투 스티커로 활용할 수 있습니다.
4. `r`을 눌러 타투를 재생성하거나, `c`를 눌러 캡쳐 후 다시 영역을 지정할 수 있습니다.


