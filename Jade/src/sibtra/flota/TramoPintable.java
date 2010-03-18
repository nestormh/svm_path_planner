public class TramoPintable
{
public TramoPintable(int xinicio,  int yinicio, int xfinal, int yfinal, double longitud)
{this.x0 = xinicio;
 this.y0= yinicio;
 this.x1= xfinal;
 this.y1= yfinal;
 this.longitud = longitud;
}

int x0, y0, x1, y1;
double longitud;

public int dimex0 ()
{return x0;
}

public int dimey0 ()
{return y0;
}

public int dimex1 ()
{return x1;
}

public int dimey1 ()
{return y1;
}

public double dimeLongitud()
{return longitud;
}

}