#include "ImageRegistration.h"

#include <dirent.h>


#define BASE_FOLDER "/home/neztol/doctorado/Datos/Canon"
#define SELECT_FOLDER false
#define DEFL_FOLDER 2
#define DFL_SIZE cvSize(320, 240)

vector<string> getFiles(const char * folder, bool folders) {
    DIR *dip;
    struct dirent *dit;
    vector<string> files;
    
    if ((dip = opendir(folder)) == NULL) {
        perror("opendir");
        return files;
    }

    while ((dit = readdir(dip)) != NULL) {
        if (dit->d_name[0] == '.') continue;
        if ((folders == true) && (dit->d_type != 0x04)) continue;
        if ((folders == false) && (dit->d_type != 0x08)) continue;
        
        string fileName(dit->d_name);
        files.push_back(fileName);
    }

    if (closedir(dip) == -1) {
        perror("closedir");
        return files;
    }

    return files;
}

void resize(IplImage * &img) {
    if ((img->width != DFL_SIZE.width) || (img->height != DFL_SIZE.height)) {
        IplImage * resized = cvCreateImage(DFL_SIZE, img->depth, img->nChannels);
        cvResize(img, resized); //, CV_INTER_CUBIC);

        cvReleaseImage(&img);
        img = resized;
    }
}

void cjtosImagenes() {
    vector<string> folders = getFiles(BASE_FOLDER, true);

    int opt = -1;
    if (SELECT_FOLDER == true) {
        cout << "Seleccione conjunto de prueba:" << endl;
        for (int i = 0; i < folders.size(); i++) {
            cout << (i + 1) << ") " << folders.at(i) << endl;
        }
        cin >> opt;

        if ((opt < 1) || (opt > folders.size())) {
            perror("Opción inválida");
            exit(-1);
        }
        opt--;
    } else {
        opt = DEFL_FOLDER;
    }

    string currentFolder(BASE_FOLDER);
    currentFolder += "/" + folders.at(opt);

    cvNamedWindow("Image1", 1);
    cvNamedWindow("Image2", 1);

    CImageRegistration registration(DFL_SIZE);

    vector<string> images = getFiles(currentFolder.c_str(), false);
    for (int i = 1; i < images.size(); i++) {
    //int i = 1;
    //int j = 2;
        string fullPath1 = currentFolder + "/" + images.at(i);
        for (int j = i + 1; j < images.size(); j++) {
            //if (i == j) continue;
            IplImage * img1 = cvLoadImage(fullPath1.c_str(), 0);
            IplImage * img1C = cvLoadImage(fullPath1.c_str(), 1);
            resize(img1);
            resize(img1C);

            cvShowImage("Image1", img1C);

            string fullPath2 = currentFolder + "/" + images.at(j);
            IplImage * img2 = cvLoadImage(fullPath2.c_str(), 0);
            IplImage * img2C = cvLoadImage(fullPath2.c_str(), 1);
            resize(img2);
            resize(img2C);

            cout << fullPath1 <<", " << fullPath2 << endl;
            cout << i << ", " << j << endl;

            cvShowImage("Image2", img2C);
            
            //pruebaSurf(img1, img2);
            /*t_moment * moments1, * moments2;
            int nMoments1, nMoments2;
            mesrTest(img1, "mser1", moments1, nMoments1);
            mesrTest(img2, "mser2", moments2, nMoments2);
            vector<t_moment *> regionPairs;
            matchMserByMoments(img1, img2, moments1, moments2, nMoments1, nMoments2, "Match", regionPairs);
            CvPoint2D32f * points1, * points2;
            int nFeat;
            cleanMatches(img1, img2, regionPairs, "Clean", points1, points2, nFeat);*/
            //starTest(img1, "star1");
            //starTest(img2, "star2");
            //registration.registration(img2, NULL, img1, NULL, NULL);
            CImageRegistration registration(cvGetSize(img1));
            registration.registration(img1, NULL, img2);

            int key = cvWaitKey(0);
            if (key == 27) exit(0);

            cvReleaseImage(&img2);
            cvReleaseImage(&img2C);

            cvReleaseImage(&img1);
            cvReleaseImage(&img1C);

        }
    }
}
