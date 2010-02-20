import java.util.*;
import java.math.*;

public class Posicion {

double x;
double y;

public Posicion()
{
 }

public Posicion(double x, double y)
{this.x = x;
 this.y = y;
 }


public void fijaPosicion(double x, double y)
{this.x = x;
 this.y = y;
}

public double dimeAngulo(Posicion destino)
{double incrementoX= destino.x - x;
 double incrementoY= destino.y - y;
 double angulo = 0;
 if (incrementoX == 0)
 {if (incrementoY > 0)
  {angulo = Math.PI/2;}
  else
  {angulo = 3*Math.PI/2;}
 }
 else{
 double tangente = incrementoY/incrementoX;
 angulo = Math.atan(tangente);}
 if (incrementoX < 0 && incrementoY <= 0)
 {angulo = angulo + Math.PI;
 }
 if (incrementoX < 0 && incrementoY > 0)
 {angulo = angulo + Math.PI;
 }
 if (incrementoX > 0 && incrementoY < 0)
 {angulo = angulo + 2*Math.PI;
 }
 return angulo;
}

public double dimeDistancia(Posicion destino)
{double incrementoX= destino.x - x;
 double incrementoY= destino.y - y;
 double distancia = incrementoX*incrementoX + incrementoY*incrementoY;
 distancia=Math.sqrt(distancia);
 return distancia;
}

}