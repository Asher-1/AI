%%
%ͼƬ����·��Ϊ��
%E:\image\car
%E:\image\person
%car��person�Ǳ��泵�����˵��ļ���
%��Щ�ļ��л������ж����
%����image�ļ��������
%�ô���������ǽ�ͼƬ���ָĳ�000123.jpg������ʽ
%%
clc;
clear;

maindir='raw_train';
name_long = 6; %ͼƬ���ֵĳ��ȣ���000123.jpgΪ6,���9λ,���޸�
num_begin = 1; %ͼ��������ʼ��������000123.jpg��ʼ�Ļ�����123

subdir = dir(maindir);
n=1;

for i = 1:length(subdir)
  if ~strcmp(subdir(i).name ,'.') && ~strcmp(subdir(i).name,'..')
     subsubdir = dir(fullfile(maindir,subdir(i).name));
    for j=1:length(subsubdir)
         if ~strcmp(subsubdir(j).name ,'.') && ~strcmp(subsubdir(j).name,'..') && ~strcmp(subsubdir(j).name,'Thumbs.db')
            img=imread(fullfile(maindir,subdir(i).name,subsubdir(j).name));
            imshow(img);
            str=num2str(num_begin,'%09d');
            newname=strcat(str,'.jpg');
            newname=newname(end-(name_long+3):end);
            system(['rename ' fullfile(maindir,subdir(i).name,subsubdir(j).name) ' ' newname]);
            num_begin=num_begin+1;
            fprintf('��ǰ�����ļ���%s',subdir(i).name);
            fprintf('�Ѿ�����%d��ͼƬ\n',n);
            n=n+1;
%            pause(0.1);%���Խ���ͣȥ��
         end
    end
  end
end