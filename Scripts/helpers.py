import numpy as np

def ask_user_0_1(question, print_1 = '', print_0 = ''):
    """
    Function to ask a binary question: 1 -> yes, 0-> no.
    
    Parameters
    ----------
    question : str
        Question that is asked
    print_1 : str, optional
        Sentence to be printed in case of answer 1
    print_0 : str, optional
        Sentence to be printed in case of answer 0
    
    Returns
    -------
    user_input : boolean
    """
    user_input = -1
    while user_input != 0 and user_input != 1:
        user_input = input(question)
        try:
            user_input = int(user_input)
            if user_input == 1:
                if print_1 != '': print(print_1)
            elif user_input == 0:
                if print_0 != '': print(print_0)
            else:
                print('Wrong input: you must type 0 or 1.')
        except:
            print('Wrong input: you must type 0 or 1.')
    return user_input


def ask_user_integer(question):
    """
    Function to ask for an integer to the user.
    
    Parameters
    ----------
    question : str
        Question that is asked

    Returns
    -------
    user_input : boolean
    """
    input_not_succeded = True
    while input_not_succeded:
        user_input = input(question)
        try:
            user_input = int(user_input)
            print(f'You have chosen {user_input}.')
            input_not_succeded = False
        except:
            print('Wrong input: you must insert an integer.')
    return user_input