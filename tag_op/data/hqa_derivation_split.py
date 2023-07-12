from copy import deepcopy
pre_dict = {'*': 3, '/': 3, '+': 2, '-': 2, '(': 1}

def is_operand(str_number):
    if (str_number.split(".")[0]).isdigit() or str_number.isdigit() or (str_number.split('-')[-1]).split(".")[
        -1].isdigit():
        return True
    else:
        return False

def make_format(operator,op1_c,op2_c):
    #op1_c = deepcopy(operand1)
    #op2_c = deepcopy(operand2)
    inst_op1 = isinstance(op1_c, list)
    inst_op2 = isinstance(op2_c, list)
    if operator == '+':
        if inst_op1:
            if op1_c[0] == '+':
                if inst_op2:
                    if op2_c[0] == '+':
                        return op1_c + op2_c[1:],True,0
                    else:
                        op1_c.append(op2_c)
                        return op1_c,True,0
                else:

                    op1_c.append(op2_c)
                    return op1_c,True,0
            else:
                if inst_op2:
                    if op2_c[0] == '+':
                        op2_c.append(op1_c)
                        return op2_c,True,0
                    else:
                        return [operator, op1_c, op2_c],False,0
                else:
                    return [operator, op1_c, op2_c],False,0
        else:
            if inst_op2:
                if op2_c[0] == '+':
                    op2_c.append(op1_c)
                    return op2_c,True,0
                else:
                    return [operator, op1_c, op2_c],False,0
            else:
                return [operator, op1_c, op2_c],False,0
    else:
        if operator == '/' and inst_op1 and op1_c[0] == '+' and inst_op2 == False and len(op1_c) - 1 == op2_c:
            op1_c[0] = 'AVERAGE'
            return op1_c,True,0
        elif operator == '-' and inst_op1 and op1_c[0] == '/' and inst_op2 == False and op2_c == 1:
            return ['-',op1_c[1],op1_c[2]], True ,1
        elif operator == '-' and inst_op2 and op2_c[0] == '/' and inst_op1 == False and op1_c == 1:
            return ['-',op2_c[1],op2_c[2]], True ,1
        else:
            return [operator, op1_c, op2_c],False,False

def infix_evaluator(infix_expression: str,avg : bool) -> int:
    '''这是中缀表达式求值的函数
    :参数 infix_expression:中缀表达式
    '''
    oplist = []
    token_list = get_token_list(infix_expression)
    #print(token_list)
    # 运算符优先级字典
    # 运算符栈
    operator_stack = []
    # 操作数栈
    operand_stack = []
    for token in token_list:
        # 数字进操作数栈
        if is_operand(token) or is_operand(token[1:]):
            operand_stack.append(float(token))
        # 左括号进运算符栈
        elif token == '(':
            operator_stack.append(token)
        # 碰到右括号，就要把栈顶的左括号上面的运算符都弹出求值
        elif token == ')':
            top = operator_stack.pop()
            while top != '(':
                # 每弹出一个运算符，就要弹出两个操作数来求值
                # 注意弹出操作数的顺序是反着的，先弹出的数是op2
                op2 = operand_stack.pop()
                op1 = operand_stack.pop()
                # 求出的值要压回操作数栈
                # 这里用到的函数get_value在下面有定义
                #operand_stack.append([top, op1, op2])
                #oplist.append([top, op1, op2])

                new_item ,issum ,per= make_format(top,op1,op2)
                operand_stack.append(new_item)

                if issum:
                    oplist[-1] = new_item
                    if per == 1:
                        oplist.append(['/',['-',op1[1],op1[2]],op1[2]])
                        operand_stack.append(['/',['-',op1[1],op1[2]],op1[2]])
                    elif per == 2:
                        oplist.append(['/', ['-', op2[1], op2[2]], op2[1]])
                        operator_stack.append(['/', ['-', op2[1], op2[2]], op2[1]])
                else:
                    oplist.append(new_item)


                # 弹出下一个栈顶运算符
                top = operator_stack.pop()
        # 碰到运算符，就要把栈顶优先级不低于它的都弹出求值
        elif token in '+-*/':
            while operator_stack and pre_dict[operator_stack[-1]] >= pre_dict[token]:
                top = operator_stack.pop()
                op2 = operand_stack.pop()
                op1 = operand_stack.pop()
                #operand_stack.append([top, op1, op2])
                #oplist.append([top, op1, op2])
                new_item ,issum,per = make_format(top, op1, op2)
                operand_stack.append(new_item)
                if issum:
                    oplist[-1] = new_item
                    if per == 1:
                        oplist.append(['/', ['-', op1[1], op1[2]], op1[2]])
                        operand_stack.append(['/', ['-', op1[1], op1[2]], op1[2]])
                    elif per == 2:
                        oplist.append(['/', ['-', op2[1], op2[2]], op2[1]])
                        operator_stack.append(['/', ['-', op2[1], op2[2]], op2[1]])
                else:
                    oplist.append(new_item)

            # 别忘了最后让当前运算符进栈
            operator_stack.append(token)
    # 表达式遍历完成后，栈里剩下的操作符也都要求值
    while operator_stack:
        top = operator_stack.pop()
        op2 = operand_stack.pop()
        op1 = operand_stack.pop()
        #operand_stack.append([top, op1, op2])
        #oplist.append([top, op1, op2])
        new_item ,issum,per = make_format(top, op1, op2)

        operand_stack.append(new_item)
        if issum:
            oplist[-1] = new_item
            if per == 1:
                oplist.append(['/', ['-', op1[1], op1[2]], op1[2]])
                operand_stack.append(['/', ['-', op1[1], op1[2]], op1[2]])
            elif per == 2:
                oplist.append(['/', ['-', op2[1], op2[2]], op2[1]])
                operator_stack.append(['/', ['-', op2[1], op2[2]], op2[1]])
        else:
            oplist.append(new_item)
    # 最后栈里只剩下一个数字，这个数字就是整个表达式最终的结果
    #print(oplist)
    return op_squence(oplist,avg)

def get_token_list(derivation):
    current_t=''
    tokens = []
    derivation=derivation.replace('[','(')
    derivation = derivation.replace(']', ')')
    derivation=derivation.replace(',','')
    derivation=derivation.replace('$', '')
    derivation = derivation.replace(' ', '')
    if 'thousand' in derivation and 'million' not in derivation and 'billion' not in derivation:
        derivation = derivation.replace('thousand','')
    if 'million' in derivation and 'thousand' not in derivation and 'billion' not in derivation:
        derivation = derivation.replace('million','')
    if 'billion' in derivation and 'million' not in derivation and 'thousand' not in derivation:
        derivation = derivation.replace('billion','')
    neg = 0
    negs = 0
    for s in derivation:
        if s == '%':
            tokens.append(str(float(current_t)*0.01))
            current_t=''
            key = 0
        elif s == '(' and current_t == '-' and len(tokens) == 0:
            tokens.append(s)
            current_t=''
            negs = 1
            neg = 0
            continue
        elif s not in pre_dict and s !=')':
            current_t += s
            key = 1

        else:
            if current_t != '':

                if negs == 1:
                    if current_t[0] == '-':
                        current_t = current_t.lstrip('-')
                    else:
                        current_t = '-'+current_t
                tokens.append(current_t)
                current_t = ''
            if s == ')':
                negs = 0
                '''
                if neg == 1:
                   neg = 0
                   continue
                '''
            if s == '-':
                if len(tokens) > 0:
                   if tokens[-1] == '(':
                      #del tokens[-1]
                      current_t += s
                      neg = 1
                      continue
                   elif tokens[-1] in pre_dict:
                       current_t += s
                       continue
                else:
                    current_t += s
                    neg = 1
                    continue
            tokens.append(s)
            current_t = ''
            key=0
    if key == 1:
        if negs == 1:
            current_t = '-' + current_t
        tokens.append(current_t)
    return tokens

def op_squence(ops,avg):
    for i in range(1, len(ops)):
        if ops[i] in ops[:i]:
            del ops[i]
    n = len(ops)
    #op_squ = deepcopy(ops)
    if n > 1:
        for i in range(1, n):
            for j in range(1, len(ops[i])):
                if isinstance(ops[i][j], list):
                    getk = False
                    for k in range(0, i):
                        if ops[k] == ops[i][j]:
                            getk = True
                    if not getk and i < n - 1:
                        if ops[i] == ops[i+1]:
                            ops[i] = ops[i][j]
                        else:
                            ops.insert(i, ops[i][j])
                    elif not getk and i == n - 1:
                        ops.insert(i, ops[i][j])
        #print(ops)
        op_squ = deepcopy(ops)
        for i in range(1, n):
            for j in range(1, len(ops[i])):
                if isinstance(ops[i][j], list):
                    for k in range(0, i):
                        if ops[k] == ops[i][j]:
                            op_squ[i][j] = str(k)
    else:
        for j in range(1, len(ops[0])):
            if isinstance(ops[0][j], list):
                ops.insert(0, ops[0][j])
        op_squ = deepcopy(ops)
        for i in range(1, len(ops)):
            for j in range(1, len(ops[i])):
                if isinstance(ops[i][j], list):
                    for k in range(0, i):
                        if ops[k] == ops[i][j]:
                            op_squ[i][j] = str(k)
    lops = len(op_squ)
    for i in range(lops):
        if op_squ[i][0] == '+':
            if len(op_squ) == 2 and (op_squ[i][1] == 1 or op_squ[i][2]) == 1:
                inc_number = op_squ[i][1] if op_squ[i][1] != 1 else op_squ[i][2]
                op_squ[i] = ["INC", inc_number]
            else:
                op_squ[i][0] = 'SUM'
        elif op_squ[i][0] == '-':
            if op_squ[i][1] == 1 and op_squ[i][2] < 1:
                op_squ[i] = ["DEC", op_squ[i][2]]
            else:
                op_squ[i][0] = 'DIFF'
        elif op_squ[i][0] == '*':
            op_squ[i][0] = 'TIMES'
        elif op_squ[i][0] == '/':
            op_squ[i][0] = 'DIVIDE'

    if avg == False:
        op_squ_out = deepcopy(op_squ)
        for i , op in enumerate(op_squ):
            if op[0] == 'AVERAGE':
                op_squ_out[i][0] = 'SUM'
                count_op = ['COUNT']+op[1:]
                op_squ_out.insert(i+1,count_op)
                op_squ_out.insert(i+2,['DIVIDE',str(i) , str(i+1)])
                if i+3 < len(op_squ_out):
                    for j in range(i+3,len(op_squ_out)):
                        for opd in range(1,len(op_squ_out[j])):
                            if isinstance(op_squ_out[j][opd],str):
                                op_squ_out[j][opd] = str(int(op_squ_out[j][opd]) + 2)
        return op_squ_out
    else:
        return op_squ

#ops = infix_evaluator('(5,940 - 598) / 598')
#print(ops)
