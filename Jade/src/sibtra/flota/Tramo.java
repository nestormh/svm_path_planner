package sibtra.flota;

import java.util.Vector;

public class Tramo {

String name;
double longitud;
int[] sucesores;
Vector vectorNombresSucesores = new Vector();


public Tramo()
{
}

public Tramo(String id)
{name = id;
}

public void fijaVectorNombresSucesores (Vector v1)
{vectorNombresSucesores = v1;
}

public Vector dimeVectorNombreSucesores()
{return vectorNombresSucesores;
}

public String dimeId()
{return name;
}

public void fijaId (String id)
{name = id;
}

public double dimeLongitud()
{return longitud;
}

public void fijaLongitud(double longitud1)
{longitud = longitud1;
}

public int[] dimeSucesores()
{return sucesores;
}

public void fijaSucesores(int[] suc)
{sucesores = suc;
}
    
}