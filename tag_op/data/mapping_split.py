from tatqa_utils import to_number

def find_table_mapping(number,table):
    for i in range(len(table)):
        row = table[i]
        for j in range(len(row)):
            cell = row[j]
            if cell == '-' or cell == '—':
                cell = '0'
            try:
                cell_number = to_number(cell)
                if abs(abs(number) - abs(cell_number)) < 0.0001 or abs(abs(number) - abs(cell_number) * 100) < 0.0001 or abs(abs(number) - abs(cell_number) * 0.01) < 0.0001:
                    return [i,j]
            except:
                continue
    return None


def find_mapping(number,table,paragraphs):
    tm = find_table_mapping(number,table)
    if tm:
        return {"table":[tm]}
    else:
        for pid in paragraphs:
            lp = len(paragraphs[pid]["text"])
            for mp1 in range(lp-1):
                for mp2 in range(mp1,lp):
                   para_number = to_number(paragraphs[pid]["text"][mp1:mp2])
                   if abs(abs(number) - abs(para_number)) < 0.0001 or abs(abs(number) - abs(para_number)*0.01) < 0.0001:
                       para_mapping = {}
                       para_mapping["paragraph"] = {}
                       para_mapping["paragraph"][pid+1] = [[mp1,mp2]]
                       return para_mapping
    return None


def split_mapping(operands,mapping,table,paras):
    temp_mapping = {}
    operand_one_mapping= {}
    operand_two_mapping = {}
    count = 0
    maped_mt = []
    for opd in operands:
        same_map = False
        maped = 0
        if isinstance(opd,str):
            if 'operator' in temp_mapping:
                temp_mapping['operator'].append(int(opd))
            else:
                temp_mapping['operator'] = [int(opd)]
            maped = 1
            if count == 0:
                if 'operator' in operand_one_mapping:
                    operand_one_mapping['operator'].append(int(opd))
                else:
                    operand_one_mapping['operator'] = [int(opd)]
            elif count == 1:
                if 'operator' in operand_two_mapping:
                    operand_two_mapping['operator'].append(int(opd))
                else:
                    operand_two_mapping['operator'] = [int(opd)]
        if maped == 0:
           if "table" in mapping:
               for mt in mapping["table"]:
                   if maped == 1:
                       break
                   cell_number = to_number(table[mt[0]][mt[1]])
                   if cell_number == None:
                       print(table)
                       print(mt)
                       print(table[mt[0]][mt[1]])
                       exit(0)
                   if (abs(abs(opd) - abs(cell_number)) < 0.0001 or abs(abs(opd) - abs(cell_number)*100) < 0.0001) and ('table' not in temp_mapping or mt not in temp_mapping['table']):
                       if mt in maped_mt:
                           print("get a same map")
                           temp_mt = mt
                           same_map = True
                       else:
                           if "table" in temp_mapping:
                               temp_mapping["table"].append(mt)
                           else:
                               temp_mapping["table"] = [mt]
                           maped = 1
                           maped_mt.append(mt)
                           if count == 0:
                               if "table" in operand_one_mapping:
                                   operand_one_mapping["table"].append(mt)
                               else:
                                   operand_one_mapping["table"] = [mt]
                           elif count == 1:
                               if "table" in operand_two_mapping:
                                   operand_two_mapping["table"].append(mt)
                               else:
                                   operand_two_mapping["table"] = [mt]
        if maped == 0:
            if "paragraph" in  mapping:
                for p in mapping["paragraph"]:
                    if maped == 1:
                        break
                    pid = int(p) - 1
                    for mp in mapping["paragraph"][p]:
                        if maped == 1:
                            break
                        para_number = to_number(paras[pid]["text"][mp[0]:mp[1]])
                        if abs(abs(opd) - abs(para_number)) < 0.0001 or abs(abs(opd) - abs(para_number)*0.01) < 0.0001:
                            if "paragraph" in temp_mapping:
                                if p in temp_mapping["paragraph"]:
                                    temp_mapping["paragraph"][p].append(mp)
                                else:
                                    temp_mapping["paragraph"][p] = [mp]
                            else:
                                temp_mapping["paragraph"] = {}
                                temp_mapping["paragraph"][p] = [mp]
                            maped = 1
                            if count == 0:
                                if "paragraph" in operand_one_mapping:
                                    if p in operand_one_mapping["paragraph"]:
                                        operand_one_mapping["paragraph"][p].append(mp)
                                    else:
                                        operand_one_mapping["paragraph"][p]= [mp]
                                else:
                                    operand_one_mapping["paragraph"] = {}
                                    operand_one_mapping["paragraph"][p] = [mp]
                            elif count == 1:
                                if "paragraph" in operand_two_mapping:
                                    if p in operand_two_mapping["paragraph"]:
                                        operand_two_mapping["paragraph"][p].append(mp)
                                    else:
                                        operand_two_mapping["paragraph"][p] = [mp]
                                else:
                                    operand_two_mapping["paragraph"] = {}
                                    operand_two_mapping["paragraph"][p] = [mp]
        if maped == 0:
            if same_map == True:
                if "table" in temp_mapping:
                    temp_mapping["table"].append(mt)
                else:
                    temp_mapping["table"] = [mt]
                maped = 1
                #maped_mt.append(mt)
                if count == 0:
                    if "table" in operand_one_mapping:
                        operand_one_mapping["table"].append(mt)
                    else:
                        operand_one_mapping["table"] = [mt]
                elif count == 1:
                    if "table" in operand_two_mapping:
                        operand_two_mapping["table"].append(mt)
                    else:
                        operand_two_mapping["table"] = [mt]
        if maped == 0 and "table" in mapping:
            map_index = find_table_mapping(opd,table)
            if map_index != None:
                if "table" in temp_mapping:
                    temp_mapping["table"].append(mt)
                else:
                    temp_mapping["table"] = [mt]
                maped = 1
                if count == 0:
                    if "table" in operand_one_mapping:
                        operand_one_mapping["table"].append(mt)
                    else:
                        operand_one_mapping["table"] = [mt]
                elif count == 1:
                    if "table" in operand_two_mapping:
                        operand_two_mapping["table"].append(mt)
                    else:
                        operand_two_mapping["table"] = [mt]
        if maped == 0:
            return None , None , None
        count += 1
    return temp_mapping , operand_one_mapping , operand_two_mapping
