// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once


#define WIN32_LEAN_AND_MEAN		// Exclude rarely-used stuff from Windows headers
#ifdef WIN32

#include <stdio.h>
#include <tchar.h>

#include <stdio.h>
#include <tchar.h>
#include <iostream>
#include <time.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <winsock2.h>
#include <cmath>
#include <algorithm>
#include <time.h>

#include <stdio.h>
#include <tchar.h>

#else

#include <stdio.h>
#include <iostream>
#include <time.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <time.h>
#include <limits.h>
#include <math.h>

#define _TCHAR char*

#endif

#include <cv.h>
#include <cvaux.h>
#include <cxcore.h>
#include <highgui.h>

#include "utils.h"

using namespace std;