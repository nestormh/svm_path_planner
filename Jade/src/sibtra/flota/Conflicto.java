package sibtra.flota;

public class Conflicto {

String tipo;
double distancia = 0;


public Conflicto(String tipo, double distancia)
{this.tipo = tipo;
 this.distancia = distancia;
}


public Conflicto()
{
}

public String dimeTipo()
{return tipo;
}

public double dimeDistancia()
{return distancia;
}

public void fijaTipo(String prioritario)
{tipo = prioritario;
}
 
public void fijaDistancia(double secundario)
{distancia = secundario;
}
   
 
}