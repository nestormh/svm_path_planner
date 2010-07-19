/* 
 * File:   CStatistics.h
 * Author: neztol
 *
 * Created on 22 de abril de 2010, 10:31
 */

#ifndef _CSTATISTICS_H
#define	_CSTATISTICS_H

#include "ViewMorphing.h"
#include "ImageRegistration.h"
#include "CMRPT_Route.h"
#include <sqlite3.h>

/*#define PATH_BASE "/home/neztol/doctorado/Datos/Estadisticas/"
 #define DB_PATH_BASE "/home/neztol/doctorado/Datos/Estadisticas/statistics.sqlite/statistics.sqlite"
#define PATH_BASE_IMG "/home/neztol/doctorado/Datos/Estadisticas/"
#define TEST_NAME "pruebasManzanilla"
#define CURRENT_SIZE cvSize(800, 600)//*/

#define BAD_ASPECT
/*#define PATH_BASE "/home/neztol/doctorado/Datos/EstadisticasITER/tripode1/"
#define DB_PATH_BASE "/home/neztol/doctorado/Datos/EstadisticasITER/statistics.sqlite"
#define PATH_BASE_IMG "/home/neztol/doctorado/Datos/EstadisticasITER/tripode1/"
#define TEST_NAME "testITERtripode1"
#define CURRENT_SIZE cvSize(800, 600)//*/



class CStatistics {
public:
    CStatistics();
    CStatistics(const CStatistics& orig);
    virtual ~CStatistics();
    //void statistics(int s, int z, int b1, int b2);
    
    void test(IplImage * imgBase, IplImage * imgRT, IplImage * imgBaseC, IplImage * imgRTC, t_Statistic_Item &item);
    void testChangingSize(string path1, string path2, CvSize currSize, t_Statistic_Item &item);
    void testChangingParams(string path1, string path2, CvSize currSize, int zoom, int b1, int b2, t_Statistic_Item &item);
    bool isThisTested(string testName, CvSize currSize, int zoom, int b1, int b2);
    void saveResults(string testName, vector<t_Statistic_Item> items);
    void statistics();
    void statistics(int testIdx, int index, int z, int s, int b1, int b2);
    void tests(int testNumber);
    void MRTP_test(CvSize size);
    void statistics(int size, int zoom, int b1, int b2);
private:
    char * PATH_BASE_STAT;
    char * DB_PATH_BASE;
    char * PATH_BASE_IMG;
    char * TEST_NAME;

};

#endif	/* _CSTATISTICS_H */

