#include <iostream>
#include <opencv.hpp>
#include <io.h>
#include <fstream>
#include <string>
#include <direct.h>
#include <conio.h>
#include <Windows.h>
#include "tinyxml2.h"

using namespace std;
using namespace cv;
using namespace tinyxml2;


class drawPos
{
public:
	explicit drawPos(string filePath)
	{
		getAllFiles(filePath);

		readAllImages();
	}


	//
	// �ֶ�����
	//
	void drawByHand()
	{
		namedWindow("drawPos");
		Mat tempImg1, tempImg2;

		for (int i = 0; i < data.size(); ++i)
		{
			system("cls");
			cout << "��" << i + 1 << "��ͼƬ\n";
			data[i].img.copyTo(tempImg1);
			for (int j = 0; j < data[i].pos.size(); ++j)
			{
				DrawRectangle(tempImg1, Rect(data[i].pos[j][0], data[i].pos[j][1], data[i].pos[j][2], data[i].pos[j][3]));
			}

			setMouseCallback("drawPos", on_MouseHandle, (void*)&tempImg1);

			bool drawFlag = false;

			while (1)
			{
				tempImg1.copyTo(tempImg2);
				switch (waitKey(30))
				{
				case 'c': // �����ͼƬ�����б����Ϣ
				{
							  data[i].pos.clear();
							  data[i].labels.clear();
							  data[i].img.copyTo(tempImg1);
							  system("cls");
							  cout << "��ͼƬ��λ���Ѿ����\n";
							  data[i].isAnnotated = false;

				}break;
				case ' ':goto end; // ��һ��
				case 'a': // ������һ��
				{
							  i--;
							  i--;
							  if (i < 0)
								  i = -1;
							  goto end;

				}
				case 'o': // ����ע��Ϣд�뵽xml
				{
							size_t cnt = 0;
							for (int j = 0; j < data.size(); ++j)
							{
								if (data[j].isAnnotated)
								{
									cnt++;

									tinyxml2::XMLDocument xmlDoc;
									XMLNode * annotation = xmlDoc.NewElement("annotation");
									xmlDoc.InsertFirstChild(annotation);

									XMLElement * pElement = xmlDoc.NewElement("folder");
									pElement->SetText("VOCType");
									annotation->InsertFirstChild(pElement);

									pElement = xmlDoc.NewElement("filename");
									pElement->SetText(data[j].name.c_str());
									annotation->InsertEndChild(pElement);

									pElement = xmlDoc.NewElement("source");
									XMLElement * pElement_sub = xmlDoc.NewElement("database");
									pElement_sub->SetText("VOC");
									pElement->InsertFirstChild(pElement_sub);
									annotation->InsertEndChild(pElement);

									pElement = xmlDoc.NewElement("size");
									pElement_sub = xmlDoc.NewElement("width");
									pElement_sub->SetText(data[j].img.size().width);
									pElement->InsertFirstChild(pElement_sub);
									pElement_sub = xmlDoc.NewElement("height");
									pElement_sub->SetText(data[j].img.size().height);
									pElement->InsertEndChild(pElement_sub);
									pElement_sub = xmlDoc.NewElement("depth");
									pElement_sub->SetText(data[j].img.channels());
									pElement->InsertEndChild(pElement_sub);
									annotation->InsertEndChild(pElement);

									pElement = xmlDoc.NewElement("segmented"); // �Ƿ�ָ�
									pElement->SetText(0);
									annotation->InsertEndChild(pElement);

									for (int k = 0; k < data[j].labels.size(); ++k)
									{
										pElement = xmlDoc.NewElement("object");
										pElement_sub = xmlDoc.NewElement("name"); // ���
										pElement_sub->SetText(data[j].labels[k].c_str());
										pElement->InsertFirstChild(pElement_sub);

										pElement_sub = xmlDoc.NewElement("pose"); // ��̬
										pElement_sub->SetText("Unspecified");
										pElement->InsertEndChild(pElement_sub);

										pElement_sub = xmlDoc.NewElement("truncated");
										pElement_sub->SetText(0);
										pElement->InsertEndChild(pElement_sub);

										pElement_sub = xmlDoc.NewElement("difficult");
										pElement_sub->SetText(0);
										pElement->InsertEndChild(pElement_sub);

										pElement_sub = xmlDoc.NewElement("bndbox");
										XMLElement* pElement_sub_sub = xmlDoc.NewElement("xmin");
										pElement_sub_sub->SetText(data[j].pos[k][0]);
										pElement_sub->InsertFirstChild(pElement_sub_sub);
										pElement_sub_sub = xmlDoc.NewElement("ymin");
										pElement_sub_sub->SetText(data[j].pos[k][1]);
										pElement_sub->InsertEndChild(pElement_sub_sub);
										pElement_sub_sub = xmlDoc.NewElement("xmax");
										pElement_sub_sub->SetText(data[j].pos[k][0] + data[j].pos[k][2]);
										pElement_sub->InsertEndChild(pElement_sub_sub);
										pElement_sub_sub = xmlDoc.NewElement("ymax");
										pElement_sub_sub->SetText(data[j].pos[k][1] + data[j].pos[k][3]);
										pElement_sub->InsertEndChild(pElement_sub_sub);
										pElement->InsertEndChild(pElement_sub);

										annotation->InsertEndChild(pElement);

									}

									string filename = "Annotations/";
									for (int x = 0; x < data[j].name.length() - 4; ++x)
									{
										filename += data[j].name[x];
									}
									filename += ".xml";
									xmlDoc.SaveFile(filename.c_str());
								}
							}
							system("cls");
							cout << "���xml�ļ����,��" << cnt << "��" << endl;
							// system("python txt.py"); // �Զ����нű�


				}break;
				} // end switch
				if (drawingBox)
				{
					DrawRectangle(tempImg2, grectangle);
					drawFlag = true;

				}
				else
				{
					if (drawFlag&&grectangle.width>5 && grectangle.height>5)
					{
						drawFlag = false;

						int *pos = new int[4];
						pos[0] = grectangle.x; pos[1] = grectangle.y;
						pos[2] = grectangle.width; pos[3] = grectangle.height;
						data[i].pos.push_back(pos);


						system("cls");

						cout << data[i].pos[data[i].pos.size() - 1][0] << "," << data[i].pos[data[i].pos.size() - 1][1] << "," << data[i].pos[data[i].pos.size() - 1][2]
							<< "," << data[i].pos[data[i].pos.size() - 1][3] << endl;
						string label("");
						cout << "�����ǩ:";
						cin >> label;
						cout << "OK,����ı�ǩ�ǣ�" << label << endl;
						data[i].labels.push_back(label);
						data[i].isAnnotated = true;

					}
				}

				imshow("drawPos", tempImg2);
			} // end while

		end:;
		}
	}
private:

	//
	// �������ͼƬ�ľ���·��
	//
	void getAllFiles(string path, string fileType = ".jpg")
	{
		// �ļ����
		long hFile = 0;
		// �ļ���Ϣ
		struct _finddata_t fileinfo;

		string p;

		if ((hFile = _findfirst(p.assign(path).append("\\*" + fileType).c_str(), &fileinfo)) != -1) {
			do {
				// �����ļ���ȫ·��
				imgPath.push_back(p.assign(path).append("\\").append(fileinfo.name));
				imgName.push_back(fileinfo.name);

			} while (_findnext(hFile, &fileinfo) == 0);  //Ѱ����һ�����ɹ�����0������-1

			_findclose(hFile);
		}
	}

	//
	// ��ȡ���е�ͼƬ
	//
	size_t readAllImages()
	{
		vector<string>::iterator begin = imgPath.begin();

		for (int i = 0; i < imgPath.size(); ++i, ++begin)
		{
			metaData temp_data;
			Mat tempImg = imread(imgPath[i]);
			if (!tempImg.empty())
			{
				temp_data.img = tempImg.clone();
				temp_data.path = imgPath[i];
				temp_data.name = imgName[i];
				data.push_back(temp_data);
			}
			else
			{
				imgPath.erase(begin);
				begin = imgPath.begin();
				begin += i;
			}
		}
		return data.size();
	}
public:

	//����
	static void DrawRectangle(cv::Mat& img, cv::Rect box)
	{
		cv::rectangle(img, Rect(box.x, box.y, box.width, box.height), Scalar(255, 100, 100), 2, 1);
	}

	//�����¼���Ӧ
	static void on_MouseHandle(int event, int x, int y, int flags, void* param)
	{
		Mat& image = *(cv::Mat*)param;

		switch (event)
		{
		case EVENT_MOUSEMOVE:
		{

								if (drawingBox)
								{
									grectangle.width = x - grectangle.x;
									grectangle.height = y - grectangle.y;
								}

		}break;
		case EVENT_LBUTTONDOWN:
		{

								  drawingBox = true;
								  grectangle = Rect(x, y, 0, 0);

		}break;
		case EVENT_LBUTTONUP:
		{

								drawingBox = false;
								if (grectangle.width < 0)
								{
									grectangle.x += grectangle.width;
									grectangle.width *= -1;
								}
								if (grectangle.height < 0)
								{
									grectangle.y += grectangle.height;
									grectangle.height *= -1;
								}


								DrawRectangle(image, grectangle);

		}break;
		default:break;

		}
	}
private:

	// �������ݽṹ
	typedef struct metaData
	{
		vector<int*> pos;
		Mat img;
		vector<string> labels;
		bool isAnnotated;
		string path;
		string name;
		metaData()
		{
			isAnnotated = false;

		}
	};

	// ���е�ͼƬ����·��
	vector<string> imgPath;
	// ����ͼƬ������
	vector<string> imgName;

	//�ṹ������
	vector<metaData> data;

	static cv::Rect grectangle;

	static bool drawingBox;
};

bool drawPos::drawingBox = false;
Rect drawPos::grectangle = Rect(-1, -1, 0, 0);



int main(int argc, char** argv)
{
	// system("python rename.py"); // �Զ����нű�
	char path[100];
	getcwd(path, 100);

	string p;
	p.append(path);
	drawPos t(p + "\\JPEGImages");
	t.drawByHand();

	return 0;
}