import java.util.*;
import java.math.*;
import java.io.*;

public class Carro {

double x;
double y;
public double orientacion;
double largo;
double ancho;
double volante;

public Carro()
{
 }

public Carro(double x, double y, double ancho, double largo, double orientacion)
{this.x = x;
 this.y = y;
 this.orientacion = orientacion;
 this.ancho = ancho;
 this.largo = largo;
 grabarDato();
 }

public Carro(Posicion position, double ancho, double largo, double orientacion)
{x = position.x;
 y= position.y;
 this.orientacion = orientacion;
this.orientacion = orientacion;
 this.ancho = ancho;
 this.largo = largo;
 grabarDato();
 }

public Posicion dimePosicion()
{Posicion posicion = new Posicion(x,y);
 return posicion;
}



public Vector datosCarro = new Vector();

public void grabarDato()
{Posicion nueva = new Posicion(x,y);
 datosCarro.addElement(nueva);
 datosCarro.addElement(String.valueOf(volante));
}

public void fijaVolante(double volante)
{this.volante=volante;
}

public void volcarDatos(String cadena)
{try
   {PrintWriter fichero = new PrintWriter(
      new BufferedWriter ( new FileWriter("carro.m")));
    fichero.println("XYalfa=["  );
   for(int i=0; i<(datosCarro.size());i=i+2)    
   {Posicion pos = (Posicion)(datosCarro.elementAt(i));
    fichero.println(pos.x + "," + pos.y + "," + (String)(datosCarro.elementAt(i+1)) + ";");
   }
    fichero.println("];");
    fichero.close();
  } catch (Exception e) {System.out.println(e);}
}


}