def ask_yes_no(prompt: str, default: str = 'y') -> bool:
    """
    선택지를 묻는 함수
    
    INPUT:
        prompt : 물어볼 내용
        default : 공백 입력시 사용할 답안
    OUTPUT:
        bool : 결과
    """


    # 입력 대소문자 통일
    default = default.lower()
    if default not in ['y', 'n']:
        raise ValueError("default must be 'y' or 'n'")

    # 공백시 사용할 답안 처리
    prompt_suffix = "((y)/n)" if default == 'y' else "(y/(n))"


    # 정상적인 답을 입력할때 까지 입력 요청
    while True:
        answer = input(f"{prompt} {prompt_suffix}: ").strip().lower()
        if answer == '':
            return default == 'y'
        elif answer in ['y', 'yes']:
            return True
        elif answer in ['n', 'no']:
            return False
        else:
            print("Please enter y or n.")



import time

def timer(label="Timer"):
    """
    이전에 이 힘수를 호출한 시점으로부터 시간을 출력
    INPUT:
        label : 설명
    """
    if not hasattr(timer, "_last_time"):
        timer._last_time = time.time()
        print(f"\n[{label}] 시작\n")
    else:
        now = time.time()
        elapsed = now - timer._last_time
        print(f"\n[{label}] 경과 시간: {elapsed:.2f}초\n")
        timer._last_time = now