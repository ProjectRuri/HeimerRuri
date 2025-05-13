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
