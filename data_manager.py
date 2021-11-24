from bs4 import BeautifulSoup
import requests
import numpy as np

class DataManager:
    def __init__(self, code):
        self.code = code

    def getMaxPage(self):
        url = 'http://finance.naver.com/item/sise_day.nhn?code={code}'.format(code=self.code)
        res = requests.get(url)
        res.encoding = 'utf-8'

        soap = BeautifulSoup(res.text, 'html.parser')
        el_table_navi = soap.find("table", class_="Nnavi")
        el_td_last = el_table_navi.find("td", class_="pgRR")
        pg_last = el_td_last.a.get('href').rsplit('&')[1]
        pg_last = pg_last.split('=')[1]
        pg_last = int(pg_last)
        return pg_last

    def getArray(self, page):
        url = 'http://finance.naver.com/item/sise_day.nhn?code={code}&page={page}'.format(code=self.code, page=page)
        res = requests.get(url)
        res.encoding = 'utf-8'

        soap = BeautifulSoup(res.text, 'html.parser')
        info_table = soap.find("table", class_="type2")

        info_lists = []
        res_lists = []
        i = 0
        for tr in info_table.find_all("tr"):
            if i == 0 or i == 1 or i == 7 or i == 8 or i == 9 or i == 15:
                i += 1
                continue
            info_list = []
            j = 0
            for td in tr.find_all("td"):
                if j == 0 or j == 2:
                    j += 1
                    continue
                info = int(td.get_text().replace(',', ''))
                info_list.append(info)
                j += 1
            info_lists.append(info_list)
            i += 1
        return info_lists
    
    def getToday(self, time_size):
        res = []
        print("data load start")
        for page in range(1, 3):
            res += self.getArray(page)
            #print("page: " + str(page) + "/" + str(max_page-1))
        print("data load finish")
        res = np.array(res)
        res = res[:time_size]
        res = res[::-1].astype('f')
        res = res.reshape(1, time_size, 5)
        return res

    def getTrainData(self):
        data = self.getAllArray()
        x_train = data[1:]
        t_train = np.zeros((x_train.shape[0], 2))
        for i in range(data.shape[0]-1):
            if data[i][0] > data[i+1][0]:
                t_train[i][0] = 0
                t_train[i][1] = 1
            else:
                t_train[i][0] = 1
                t_train[i][1] = 0
        #i = np.argmax(x_train[:, 0])
        #x_standard = x_train[:, 0].reshape((x_train.shape[0], -1))
        #x_train[:, :4] = x_train[:, :4] / x_train[i, 0] * 100
        #x_train[:, 4] = x_train[:, 4] / x_train[i, 4]
        x_max = np.max(x_train, axis=0)
        x_min = np.min(x_train, axis=0)
        x_train = (x_train.astype('f') - x_min) / (x_max - x_min)
        x_train = x_train[::-1] 
        print(x_train)
        t_train = t_train.astype('f')
        t_train = t_train[::-1]
        return x_train, t_train

    def getAllArray(self):
        res = []
        max_page = self.getMaxPage()
        print("data load start")
        for page in range(1, max_page):
            res += self.getArray(page)
            print("page: " + str(page) + "/" + str(max_page-1))
        print("data load finish")
        res = np.array(res)
        return res
        
    def getData(self, time_size):
        data = self.getAllArray()
        x = data[1:].astype('f')
        t = np.zeros((x.shape[0], 2)).astype('f')
        for i in range(data.shape[0]-1):
            if data[i][0] > data[i+1][0]:
                t[i][0] = 0
                t[i][1] = 1
            else:
                t[i][0] = 1
                t[i][1] = 0
        x, t = self.volume(x, t, time_size)
        line = x.shape[0] - 100
        x_train, t_train = x[:line], t[:line]
        x_test, t_test = x[line:], t[line:]
        return (x_train, t_train), (x_test, t_test)

    def volume(self, x, t, time_size):
        iters = x.shape[0] - time_size + 1
        x_res = self.normalize(x[:time_size])
        x_res = x_res[::-1]
        for i in range(1,iters):
            x_slice = self.normalize(x[i:i+time_size])
            x_slice = x_slice[::-1]
            x_res = np.concatenate((x_res, x_slice), axis=0)
        x_res = x_res.reshape((-1, time_size, x.shape[1]))
        t_res = t[:iters]
        x_res, t_res = x_res[::-1], t_res[::-1]
        return x_res, t_res

    def normalize(self, x):
        x_max = np.array([np.max(x[:,:4]), np.max(x[:,:4]), np.max(x[:,:4]), np.max(x[:,:4]), np.max(x[:,4])])
        x_min = np.array([np.min(x[:,:4]), np.min(x[:,:4]), np.min(x[:,:4]), np.min(x[:,:4]), np.min(x[:,4])])
        x = (x - x_min) / (x_max - x_min)
        return x

    def normalize2(self, x):
        x_max = np.max(x, axis=0)
        x_min = np.min(x, axis=0)
        x = (x - x_min) / (x_max - x_min)
        return x , x_max[0], x_min[0]

    def getData2(self, time_size):
        data = self.getAllArray()
        data = data[::-1].astype('f')
        data, x_max, x_min = self.normalize2(data)
        x = self.volume2(data[:data.shape[0]-1], time_size)
        t = data[time_size:, 0]
        line = x.shape[0]-100
        x_train = x[:line]
        t_train = t[:line]
        x_test = x[line:]
        t_test = t[line:]
        
        print("x_train: {x_train}, t_train: {t_train}, x_test: {x_test}, t_test: {t_test}".format(x_train=x_train.shape, t_train=t_train.shape,
        x_test=x_test.shape, t_test=t_test.shape))

        return (x_train, t_train), (x_test, t_test), (x_max, x_min)

    def volume2(self, x, time_size):
        iters = x.shape[0] - time_size + 1
        x_res = x[:time_size]
        for i in range(1,iters):
            x_res = np.concatenate((x_res, x[i:i+time_size]), axis=0)
        x_res = x_res.reshape((-1, time_size, x.shape[1]))
        return x_res

    def getData3(self, time_size):
        data = self.getAllArray()
        data = data[::-1].astype('f')
        #data, x_max, x_min = self.normalize2(data)

        x_train = self.volume2(data[:data.shape[0]-1], time_size)
        t_train = data[time_size:, 0]
        x_today = data[data.shape[0]-time_size:]
        x_today = x_today.reshape((1, time_size, 5))
        print("x_train: {x_shape}, t_train: {t_shape}".format(x_shape=x_train.shape, t_shape=t_train.shape))

        return x_train, t_train