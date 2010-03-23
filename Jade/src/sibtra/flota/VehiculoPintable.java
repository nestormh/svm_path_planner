package sibtra.flota;

public class VehiculoPintable
{
public VehiculoPintable(int xinicio,  int yinicio)
{this.x0 = xinicio;
 this.y0= yinicio;

 
}

int x0, y0, x1, y1;
double longitud;

public int dimex0 ()
{return x0;
}

public int dimey0 ()
{return y0;
}

public void fijax0 (int x0)
{this.x0= x0;
}

public void fijay0 (int y0)
{this.y0= y0;
}

public int dimex1 ()
{return x1;
}

public int dimey1 ()
{return y1;
}


}