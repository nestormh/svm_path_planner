public class Vehiculo {

String name;
double velocidad;
double longitudTramo;
double longitudEnTramo;
String tramo;
String proximoTramo ="";
String estado = "Normal";
String posicionVehiculo;
String[] rutas;
boolean rutaCircular = true;

public Vehiculo()
{
}

public Vehiculo(String id)
{name = id;
}

public String dimeSiguienteTramo()
{String actual = tramo;
 for (int i=0; i < rutas.length; i++)
 {if (actual.equals(rutas[i]))
  {if (i < rutas.length - 1 ) 
   {proximoTramo = rutas[i+1];
   }
   else
   {proximoTramo = " ";
   }
  }
 }
 return proximoTramo;
}

public void fijaRuta(String[] ruta)
{rutas = ruta;
}

public String[] dimeRuta()
{return rutas;
}

public String dimeTramoEnRuta(int i)
{return rutas[i];
}

public String dimeId()
{return name;
}

public String dimePosicionVehiculo()
{return posicionVehiculo;
}

public void fijaPosicionVehiculo(String posicion)
{posicionVehiculo = posicion;
}

public void fijaId (String id)
{name = id;
}

public String dimeEstado()
{return estado;
}

public String dimeTramo()
{return tramo;
}

public String dimeProximoTramo()
{return proximoTramo;
}

public double dimeVelocidad()
{if (dimeEstado().equals("EnEspera"))
 {return 0;
 }
 else
 {return velocidad;}
}

public double dimeLongitudTramo()
{return longitudTramo;
}

public double dimeLongitudEnTramo()
{return longitudEnTramo;
}

public void fijaEstado(String estado1)
{estado = estado1;
}

public void  fijaTramo(String tramo2)
{tramo = tramo2;
}

public void  fijaProximoTramo(String proximoTramo2)
{proximoTramo = proximoTramo2;
}

public void  fijaVelocidad(double int1)
{velocidad = int1;
}

public void  fijaLongitudTramo(double int1)
{longitudTramo = int1;
}

public void fijaLongitudEnTramo(double int1)
{longitudEnTramo = int1;
}
    
}