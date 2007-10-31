package carrito.gps;

public class Prueba {
  private static double p[] = { 5105973.3468139370f, -3590022.188114373f, -1310555.0643268004f };
  private static double q[] = { 5106047.3157405285f, -3589914.109460398f, -1310510.0003116030f };
  private static double r[] = { 5105943.1079750445f, -3589986.3739085793f, -1310730.8419269559f };


  public static void setParams() {
    double a[] = { q[0] - p[0], q[1] - p[1], q[2] - p[2] };
    double b[] = { r[0] - p[0], r[1] - p[1], r[2] - p[2] };

    double x[] = a;
    double z[] = {
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    };
    double res = x[0] * z[0] + x[1] * z[1] + x[2] * z[2];
    System.out.println(res);
    double y[] = {
        x[1] * z[2] - x[2] * z[1],
        x[2] * z[0] - x[0] * z[2],
        x[0] * z[1] - x[1] * z[0]
    };
    res = x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
    System.out.println(res);
    res = z[0] * y[0] + z[1] * y[1] + z[2] * y[2];
    System.out.println(res);

    double kx = Math.sqrt(Math.pow(x[0], 2) + Math.pow(x[1], 2) + Math.pow(x[2], 2));
    double ky = Math.sqrt(Math.pow(y[0], 2) + Math.pow(y[1], 2) + Math.pow(y[2], 2));
    double kz = Math.sqrt(Math.pow(z[0], 2) + Math.pow(z[1], 2) + Math.pow(z[2], 2));

    for (int i = 0; i < 3; i++) {
      x[i] /= kx;
      y[i] /= ky;
      z[i] /= kz;
    }

    System.out.println("hold on");
    System.out.println("plot3([" + p[0] + ", " + q[0] + "], [" + p[1] + ", " + q[1] + "], [" + p[2] + ", " + q[2] + "])");
    System.out.println("plot3([" + p[0] + ", " + r[0] + "], [" + p[1] + ", " + r[1] + "], [" + p[2] + ", " + r[2] + "])");
    System.out.println("plot3([0, " + a[0] + "], [0, " + a[1] + "], [0, " + a[2] + "])");
    System.out.println("plot3([0, " + b[0] + "], [0, " + b[1] + "], [0, " + b[2] + "])");
    System.out.println("plot3([0, " + x[0] + "], [0, " + x[1] + "], [0, " + x[2] + "])");
    System.out.println("plot3([0, " + y[0] + "], [0, " + y[1] + "], [0, " + y[2] + "])");
    System.out.println("plot3([0, " + z[0] + "], [0, " + z[1] + "], [0, " + z[2] + "])");
    System.out.println("hold off");

  }

  public static void main(String[] args) {
    Prueba.setParams();
  }
}
