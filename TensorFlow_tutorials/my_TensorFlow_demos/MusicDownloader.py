#! python3
# -*- coding: utf-8 -*-
# QQ音乐下载器(MusicDownloader.py)
# 版本（version）   1.0
# 日期（Date）  2018/01/13
import requests
import os

class MusicDownloader():
    '''
    QQ音乐下载器
    '''

    def search(self, song_name):
        '''搜索音乐'''
        headers = {'referer':'http://y.qq.com/portal/search.html',
                   'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.84 Safari/537.36'
                   }
        query_parameters = {'ct':'24',
                            'qqmusic_ver':'1298',
                            'new_json':'1',
                            'remoteplace':'txt.yqq.center',
                            'searchid':'48987344780034770',
                            't':'0',
                            'aggr':'1',
                            'cr':'1',
                            'catZhida':'1',
                            'lossless':'0',
                            'flag_qc':'0',
                            'p':'1', #分页
                            'n':'20', #显示的结果数量
                            'w':str(song_name), #必填搜索歌名
                            'g_tk':'5381',
                            'jsonpCallback':'MusicJsonCallback8319472269210575',
                            'loginUin':'0',
                            'hostUin':'0',
                            'format':'json', #json或者jsonp
                            'inCharset':'utf8',
                            'outCharset':'utf-8',
                            'notice':'0',
                            'platform':'yqq',
                            'needNewCode':'0'
                            }
        search_url = 'http://c.y.qq.com/soso/fcgi-bin/client_search_cp'

        req = requests.get(url=search_url, params=query_parameters, headers=headers)
        req.raise_for_status()
        req_json = req.json()
        song_list = list(req_json.get('data').get('song').get('list'))

        result_list = []
        for song in song_list:
            result = {}
            result['sid'] = song.get('id')
            result['mid'] = song.get('mid')
            result['media_mid'] = song.get('file').get('media_mid')
            result['song_title'] = song.get('title') #歌名
            result['album_title'] = song.get('album').get('title') #专辑名称
            result['singer_name'] = song.get('singer')[0].get('name') #歌手
            result['interval'] = song.get('interval') #时长
            result_list.append(result)

        self.print_song_list(result_list)
        req.close()
        return result_list

    def get_song_url(self, song):
        '''获取歌曲'''
        headers = {'referer':'http://y.qq.com/portal/search.html',
                   'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.84 Safari/537.36'
                   }
        query_parameters = {'g_tk':'5381',
                            # 'jsonpCallback':'MusicJsonCallback8064462801977872',
                            'loginUin':'0',
                            'hostUin':'0',
                            'format':'json',
                            'inCharset':'utf8',
                            'outCharset':'utf-8',
                            'notice':'0',
                            'platform':'yqq',
                            'needNewCode':'0',
                            'cid':'205361747',
                            # 'callback':'MusicJsonCallback8064462801977872', #注释可以关闭jsonp
                            'uin':'0',
                            'songmid':str(song.get('mid')),
                            'filename':'C400'+str(song.get('media_mid'))+'.m4a',
                            'guid':'6179861260'
                            }
        song_url = 'http://c.y.qq.com/base/fcgi-bin/fcg_music_express_mobile3.fcg'

        req = requests.get(url=song_url, params=query_parameters, headers=headers)
        req.raise_for_status()
        req_json = req.json()
        item = req_json.get('data').get('items')[0]
        req.close()

        return item

    def download(self, item, song):
        '''
        下载歌曲
        :param item:
        :return:
        '''
        root = os.path.abspath('.')
        save_dir = os.path.join(root,'music')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        headers = {'referer':'http://y.qq.com/portal/search.html',
                   'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.84 Safari/537.36'
                   }
        query_parameters = {'guid':'6179861260',
                            'vkey':item.get('vkey'),
                            'uin':'',
                            'fromtag':'999'
                            }
        download_url = 'http://dl.stream.qqmusic.qq.com/'+item.get('filename')

        with requests.get(url=download_url, params=query_parameters, headers=headers, stream=True) as req:
            req.raise_for_status()
            chunk_size = 1024
            count = 0
            total_size = int(req.headers['content-length'])
            print('>>>歌曲：{} | 歌手：{}\t开始下载↓'.format(song.get('song_title'),song.get('singer_name')))
            with open(os.path.join(save_dir, song.get('song_title')+'.m4a'), 'wb') as music_file:
                for music in req.iter_content(chunk_size=chunk_size):
                    music_file.write(music)
                    count = count + 1
                    self.schedule(count, chunk_size, total_size)

        music_file.close()
        req.close()


    def schedule(self, block_num, block_size, total_size):
        '''
        下载进度
        :param block_num:
        :param block_size:
        :param total_size:
        :return:
        '''
        per = 100.0 * block_num * block_size / total_size
        end_str = "\r"
        if per > 100:
            per = 100
            end_str = '\n'
        print('下载进度：%s%d%%'%('■'*int(per/2),per), end=end_str)

    def print_song_list(self, song_list):
        '''
        格式输出歌曲列表
        :param song_list:
        :return:
        '''
        print('\n编号\t|歌曲\t|歌手\t|时长\t|专辑')
        print('----------------------------------')
        for key, song in enumerate(song_list):
            mm = int(song.get('interval')) // 60
            mm = str(mm) if mm >= 10 else '0'+str(mm)
            ss = int(song.get('interval')) % 60
            ss = str(ss) if ss >= 10 else '0'+str(ss)
            print('{}\t|{}\t|{}\t|{}\t|{}\t'.format(
                key,
                song.get('song_title'),
                song.get('singer_name'),
                mm+':'+ss,
                song.get('album_title')
            ))
        print('----------------------------------\n')

    def start_download(self):
        '''开始下载'''
        print('■■■■■■■■■■■■■■■■■■■■')
        print('\n            QQ音乐下载器\n')
        print('■■■■■■■■■■■■■■■■■■■■')
        host = 'http://y.qq.com/'
        headers = {'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.84 Safari/537.36'}
        try:
            req = requests.get(url=host,headers=headers)
            req.raise_for_status()
            req.close()
        except:
            print('无法访问y.qq.com，请检测是否联网或网址是否有效！！！')
            return None

        while True:
            try:
                song_name = input('\n> 搜索（请输入歌名）：')
                search_result = self.search(song_name)
            except Exception as e:
                print(e)
                print('歌曲搜索失败！！！请结束下载')
            try:
                num = int(input('> 请输入歌曲编号：'))
                song_result = self.get_song_url(search_result[num])
            except Exception as e:
                print(e)
                print('歌曲解析失败！！！请结束下载')
            try:
                self.download(song_result, search_result[num])
            except Exception as e:
                print(e)
                print('歌曲下载失败！！！请结束下载')

            is_continue = input('是否继续下载（y/n）：')
            if is_continue.lower() == 'n':
                break;
        print('>>>退出下载')

if __name__=='__main__':
    md = MusicDownloader()
    md.start_download()
