package carrito.gps;

public class Geometria {

  public static double getAngulo(double u[], double v[], int eje, int sentido) {
    /*double mod_u = Math.sqrt(Math.pow(u[0], 2) + Math.pow(u[1], 2));
    double mod_v = Math.sqrt(Math.pow(v[0], 2) + Math.pow(v[1], 2));
    double divisor = mod_u * mod_v;
    double suma = u[0] * v[0] + u[1] * v[1];
    if (divisor == 0) {
      divisor += 0.00000000000000001f;
    }*/

  double mod_u = 0.0;
  double mod_v = 0.0;
  for (int i = 0; i < 3; i++) {
    if (i == eje)
      continue;
    System.out.print(u[i] + ", ");
  }
  System.out.println();

  for (int i = 0; i < 3; i++) {
    if (i == eje)
      continue;
    System.out.print(v[i] + ", ");
  }
  System.out.println();

  double suma = 0;
  for (int i = 0; i < 3; i++) {
    if (i == eje)
      continue;
    mod_u += Math.pow(u[i], 2);
    mod_v += Math.pow(v[i], 2);
    suma += u[i] * v[i];
  }
  mod_u = Math.sqrt(mod_u);
  mod_v = Math.sqrt(mod_v);

System.out.println();
  if (v[sentido] < 0)
    return Math.acos(suma / (mod_u * mod_v));
  else
    return -Math.acos(suma / (mod_u * mod_v));
  }

  public static boolean corta(double r1[], double r2[]) {
    double m1 = (r1[3] - r1[1]) / (r1[2] - r1[0]);
    double m2 = (r2[3] - r2[1]) / (r2[2] - r2[0]);
    double n1 = (r1[1] * r1[2] - r1[3] * r1[0]) / (r1[2] - r1[0]);
    double n2 = (r2[1] * r2[2] - r2[3] * r2[0]) / (r2[2] - r2[0]);

    double x = (n2 - n1);
    if (m1 != m2) {
      x /= (m1 - m2);
    }
    else {
      return false;
    }

    if ( (x > Math.max(r1[0], r1[2])) || (x < Math.min(r1[0], r1[2]))) {
      return false;
    }
    if ( (x > Math.max(r2[0], r2[2])) || (x < Math.min(r2[0], r2[2]))) {
      return false;
    }
    double y = m1 * x + n1;
    if ( (y > Math.max(r1[1], r1[3])) || (y < Math.min(r1[1], r1[3]))) {
      return false;
    }
    if ( (y > Math.max(r2[1], r2[3])) || (y < Math.min(r2[1], r2[3]))) {
      return false;
    }

    return true;

  }

  public static boolean enPoligono(double punto[], double poligono[]) {
    int cortes = 0;
    for (int i = 2; i < poligono.length; i += 2) {
      double r1[] = {
          punto[0], punto[1], 9999999999999.99f, punto[1]};
      double r2[] = {
          poligono[i - 2], poligono[i - 1], poligono[i], poligono[i + 1]};
      if ( (corta(r1, r2)) && (poligono[i + 1] != punto[1])) {
        System.out.print("(" + poligono[i - 2] + ", " + poligono[i - 1] +
                         ") --- ");
        System.out.println("(" + poligono[i] + ", " + poligono[i + 1] + ")");
        cortes++;
      }
    }

    if (cortes % 2 == 0) {
      return false;
    }
    else {
      return true;
    }
  }

}
