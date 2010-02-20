import java.math.*;

public class Recta {

/* ax + by + c = 0 */

Posicion posicionInicio;
Posicion posicionFinal;
double a;
double b;
double c;


public Recta()
{
 }

public Recta(double a, double b, double c)
{this.a = a;
 this.b = b;
 this.c = c;
}

public Recta(Posicion inicio, Posicion meta)
{fijaPosicionInicio(inicio);
 fijaPosicionFinal(meta);
 if (inicio.x == meta.x)
 {a = 1;
  b = 0;
  c= -inicio.x;
 }
 if (inicio.y == meta.y)
 {a = 0;
  b= 1;
  c = -inicio.y;
 }
 if ( (inicio.y != meta.y) && (inicio.x != meta.x) )
  { a = 1/(meta.x - inicio.x);
    b = -1/(meta.y - inicio.y);
    c = -a*inicio.x - b*inicio.y;
  }
}

public double dimeA()
{return a;
}

public double dimeB()
{return b;
}

public double dimeC()
{return c;
}

/* 1 para positivo, 0 para negativo) */
public int dimeSigno(Posicion punto)
{int signo = 0;
 double calculo = a * (punto.x - posicionInicio.x) + b * (punto.y - posicionInicio.y);
 if (calculo > 0)
  {signo = 1;
  }
 if (calculo < 0)
  {signo = 0;
  }
 return signo;
}

public void fijaPosicionInicio(Posicion inicio)
{posicionInicio = inicio;
}

public void fijaPosicionFinal(Posicion posFinal)
{posicionFinal = posFinal;
}

public void fijaPosiciones(Posicion inicial, Posicion posFinal)
{fijaPosicionInicio(inicial);
 fijaPosicionFinal(posFinal);
}

public Posicion dimeCorte (Recta otra)
{double D = otra.dimeA();
 double E = otra.dimeB();
 double F = otra.dimeC();
 Posicion corte = new Posicion(-100000, -1000000);
 if ( (a*E - b*D) != 0)
 {double xcorte = (-c*E + F*b)/(a*E - b*D);
  double ycorte = (-a*F + c*D)/(a*E - b*D);
  corte = new Posicion(xcorte, ycorte);
 }
 return corte;
}

}