#ifndef _FILE_LIST_H
#define _FILE_LIST_H

#include <string.h>
#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

typedef struct {
    boost::posix_time::time_duration timestamp;
    std::string LW;
    std::string LS;
    std::string RW;
    std::string RS;    
} t_ImageNames;

#endif