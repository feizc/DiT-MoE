import json 


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def heatmap_plt(static_list):
    import numpy as np
    import matplotlib.pyplot as plt 
    y = [x * 100 for x in range(10)] 
    y1 = [x * 10 for x in range(10)]

    for i in range(12): 
        data = [x[i] for x in static_list] 
        data = np.stack(data, axis=0) 
        # data = normalization(data)
        plt.subplot(2, 6, i + 1)
        plt.imshow(data, cmap='Oranges', aspect='auto') 
        if i == 0 or i==6:
            plt.ylabel('Image classes')
        plt.title('MoE Layer ' + str(i))
        if i < 6:
            plt.xticks([]) 
        if i== 0 or i == 6:
            plt.yticks(y1, y)
        else:
            plt.yticks([])

    # plt.colorbar()  # 添加颜色条 
    # plt.title('Value Heatmap Example')
    # plt.xlabel('Expert ids')
    # plt.ylabel('Image classes')
    plt.subplots_adjust(wspace=0.1)
    plt.show() 


import numpy as np 

def static_for_class(data, layer_num=12): 
    static_list = np.zeros((layer_num, 8)) # [[0] * 8] * layer_num
    for data_list in data: 
        for j in range(3000):
            row = j % layer_num 
            for k in range(256): 
                col = k % 8 
                expert_id = data_list[j][k][0]
                expert_id2 = data_list[j][k][1]
                static_list[row][expert_id] += 1
                static_list[row][expert_id2] += 1
    return static_list 


import os 
calculate_flag = False 
static_list = [] 

for i in range(0, 1000):
    if i % 10 != 0:
        continue 

    path = os.path.join('data/experts', str(i) + '.json') 
    tgt_path = os.path.join('data/class', str(i) + '.npy') 

    if calculate_flag == True: 
        with open(path, 'r') as f: 
            data_list = json.load(f) 
        # print(len(data_list), len(data_list[0]), len(data_list[0][0])) 
        # 50, 3000 (250 * 12), 512 (256 * 2), 2 
        # print(data_list[0][0][0])
        # print(data_list[0][1][0])
        # print(data_list[0][2][0])  
        # continue
        static = static_for_class(data_list) 
        np.save(tgt_path, static)
    else:
        static = np.load(tgt_path)

    print(i)
    print(static)
    static_list.append(static)
    # print(static_list)


heatmap_plt(static_list)

