#include "svmpathplanning.h"

int main ( int argc, char * argv[] )
{

    SVMPathPlanning pathPlanner;
    
    pathPlanner.testSingleProblem();

    return 0;
}

int main2(int argc, char **argv)
{
//     int i;
//     char input_file_name[1024];
//     char model_file_name[1024];
//     const char *error_msg;
//     
//     parse_command_line(argc, argv, input_file_name, model_file_name);
//     read_problem(input_file_name);
//     error_msg = svm_check_parameter(&prob,&param);
//     if(error_msg)
//     {
//         fprintf(stderr,"Error: %s\n",error_msg);
//         exit(1);
//     }
//     
//     if(cross_validation)
//     {
//         do_cross_validation_with_KM_precalculated(  );
//         
//         //      do_cross_validation();
//     }
//     else
//     {
//         model = svm_train(&prob,&param);
//         if(svm_save_model(model_file_name,model))
//         {
//             fprintf(stderr, "can't save model to file %s\n", model_file_name);
//             exit(1);
//         }
//         svm_free_and_destroy_model(&model);
//     }
//     svm_destroy_param(&param);
//     free(prob.y);
//     
//     #ifdef _DENSE_REP
//     for (i = 0; i < prob.l; ++i)
//         free((prob.x+i)->values);
//     #else
//         free(x_space);
//         #endif
//         free(prob.x);
//         free(line);
//         
//         return 0;
}