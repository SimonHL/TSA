
def inner_func():
    #global a
    a = 5
    print a
    

def main_test():
    a = 10
    inner_func()
    print a


    
if __name__ == '__main__':
    a = 10
    inner_func()
    print a