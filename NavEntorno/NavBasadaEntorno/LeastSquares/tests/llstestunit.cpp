
#include <stdafx.h>
#include <stdio.h>
#include "llstestunit.h"

bool testlls(bool silent)
{
    bool result;
    bool waserrors;
    bool glserrors;
    bool plserrors;
    bool linlserrors;
    double threshold;
    int n;
    int m;
    int maxn;
    int maxm;
    int i;
    int j;
    int k;
    int pass;
    int passcount;
    int ctask;
    double xscale;
    ap::real_1d_array x;
    ap::real_1d_array y;
    ap::real_1d_array w;
    ap::real_1d_array c;
    ap::real_1d_array ac;
    double c1;
    double c2;
    double v;
    double s1;
    double s2;
    double s3;
    double delta;
    double t;
    double vg;
    double vk;
    double vk1;
    double vk2;
    double vp;
    ap::real_2d_array a;

    waserrors = false;
    glserrors = false;
    plserrors = false;
    linlserrors = false;
    threshold = 10000*ap::machineepsilon;
    maxn = 10;
    maxm = 5;
    passcount = 10;
    delta = 0.001;
    
    //
    // Testing general least squares
    //
    for(n = 1; n <= maxn; n++)
    {
        x.setbounds(0, n-1);
        y.setbounds(0, n-1);
        w.setbounds(0, n-1);
        ac.setbounds(0, n-1);
        for(m = 1; m <= maxm; m++)
        {
            a.setbounds(0, n-1, 0, m-1);
            for(pass = 1; pass <= passcount; pass++)
            {
                
                //
                // Prepare task.
                // Use Chebyshev basis. Its condition number is very good.
                //
                xscale = 0.9+0.1*ap::randomreal();
                for(i = 0; i <= n-1; i++)
                {
                    if( n==1 )
                    {
                        x(i) = 2*ap::randomreal()-1;
                    }
                    else
                    {
                        x(i) = xscale*(double(2*i)/double(n-1)-1);
                    }
                    y(i) = 3*x(i)+exp(x(i));
                    w(i) = 1+ap::randomreal();
                    a(i,0) = 1;
                    if( m>1 )
                    {
                        a(i,1) = x(i);
                    }
                    for(j = 2; j <= m-1; j++)
                    {
                        a(i,j) = 2*x(i)*a(i,j-1)-a(i,j-2);
                    }
                }
                
                //
                // Solve General Least Squares task
                //
                buildgeneralleastsquares(y, w, a, n, m, c);
                glserrors = glserrors||!isglssolution(n, m, y, w, a, c);
            }
        }
    }
    
    //
    // Test polynomial least squares
    //
    for(n = 1; n <= maxn; n++)
    {
        x.setbounds(0, n-1);
        y.setbounds(0, n-1);
        w.setbounds(0, n-1);
        ac.setbounds(0, n-1);
        for(m = 1; m <= maxm; m++)
        {
            a.setbounds(0, n-1, 0, m-1);
            for(pass = 1; pass <= passcount; pass++)
            {
                
                //
                // Prepare task.
                // Use power basis.
                //
                xscale = 0.9+0.1*ap::randomreal();
                for(i = 0; i <= n-1; i++)
                {
                    if( n==1 )
                    {
                        x(i) = 2*ap::randomreal()-1;
                    }
                    else
                    {
                        x(i) = xscale*(double(2*i)/double(n-1)-1);
                    }
                    y(i) = 3*x(i)+exp(x(i));
                    w(i) = 1;
                    a(i,0) = 1;
                    for(j = 1; j <= m-1; j++)
                    {
                        a(i,j) = x(i)*a(i,j-1);
                    }
                }
                
                //
                // Solve polynomial least squares task
                //
                buildpolynomialleastsquares(x, y, n, m-1, c);
                plserrors = plserrors||!isglssolution(n, m, y, w, a, c);
            }
        }
    }
    
    //
    // Linear approximation.
    // These tests are easy to do, but I think it will be enough
    //
    for(n = 1; n <= maxn; n++)
    {
        x.setbounds(0, 2*n-1);
        y.setbounds(0, 2*n-1);
        for(pass = 1; pass <= passcount; pass++)
        {
            
            //
            // Generate y = C1 + C2*x
            // Generate N pairs of points: (x, y(x)+s1) and (x, y(x)-s1)
            // C1 and C2 must be calculated exactly
            //
            c1 = 2*ap::randomreal()-1;
            c2 = 2*ap::randomreal()-1;
            s1 = 1;
            for(i = 0; i <= n-1; i++)
            {
            }
        }
    }
    
    //
    //
    // report
    //
    waserrors = glserrors||plserrors;
    if( !silent )
    {
        printf("TESTING LINEAR LEAST SQUARES\n");
        printf("GENERAL LLS                              ");
        if( glserrors )
        {
            printf("FAILED\n");
        }
        else
        {
            printf("OK\n");
        }
        printf("POLYNOMIAL LLS                           ");
        if( plserrors )
        {
            printf("FAILED\n");
        }
        else
        {
            printf("OK\n");
        }
        if( waserrors )
        {
            printf("TEST FAILED\n");
        }
        else
        {
            printf("TEST PASSED\n");
        }
        printf("\n\n");
    }
    
    //
    // end
    //
    result = !waserrors;
    return result;
}


bool isglssolution(int n,
     int m,
     const ap::real_1d_array& y,
     const ap::real_1d_array& w,
     const ap::real_2d_array& fmatrix,
     const ap::real_1d_array& c)
{
    bool result;
    int i;
    int j;
    ap::real_1d_array ac;
    double v;
    double s1;
    double s2;
    double s3;
    double delta;

    delta = 0.001;
    ac.setbounds(0, n-1);
    
    //
    // Test result
    //
    result = true;
    for(i = 0; i <= n-1; i++)
    {
        v = ap::vdotproduct(&fmatrix(i, 0), &c(0), ap::vlen(0,m-1));
        ac(i) = v;
    }
    s1 = 0;
    for(i = 0; i <= n-1; i++)
    {
        s1 = s1+ap::sqr(w(i)*(ac(i)-y(i)));
    }
    for(j = 0; j <= m-1; j++)
    {
        s2 = 0;
        s3 = 0;
        for(i = 0; i <= n-1; i++)
        {
            s2 = s2+ap::sqr(w(i)*(ac(i)+fmatrix(i,j)*delta-y(i)));
            s3 = s3+ap::sqr(w(i)*(ac(i)-fmatrix(i,j)*delta-y(i)));
        }
        result = result&&s2>=s1&&s3>=s1;
    }
    return result;
}


/*************************************************************************
Silent unit test
*************************************************************************/
bool llstestunit_test_silent()
{
    bool result;

    result = testlls(true);
    return result;
}


/*************************************************************************
Unit test
*************************************************************************/
bool llstestunit_test()
{
    bool result;

    result = testlls(false);
    return result;
}



