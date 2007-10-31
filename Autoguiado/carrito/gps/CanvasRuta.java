package carrito.gps;


import java.util.*;

import java.awt.*;
import java.awt.event.*;
import carrito.server.serial.ControlCarro;

public class CanvasRuta extends Canvas implements MouseListener, KeyListener {
  public int ancho = 320;
  public int alto = 240;

  public double rotacion = Math.PI / 2;

  private int origenX = 0;
  private int origenY = 0;

  private static final long serialVersionUID = 10;

  private Point ruta[] = null;
  private Point p = new Point(5,5);
  private int cercano = 0;
  private Point vectores[] = null;
  private double angulos[] = null;
  private double velocidades[] = null;

  // Ruta seguida a posteriori
  private Point ruta2[] = null;
  private Point vectores2[] = null;
  private double angulos2[] = null;
  private double velocidades2[] = null;

  private double aumento = 1;

  private double simulaVel = 10;

  private double miAngulo = Math.PI / 2;
  private double miVelocidad = 20.0f;
  private Point miVector = new Point(0, 0);
  private Point estima = new Point(0,0);
  private double giroRuedas = 0;
  private double velEstimada = 0;

  private Vector trayectoria = new Vector();
  private Vector obstaculos = new Vector();
  private double minX = 0, minY = 0, maxX = ancho, maxY = alto;
  private double kx = 0, ky = 0;

  private CambioCoordenadas cc = null;

  private double anguloRuedas = ControlCarro.CARRO_CENTRO + ControlCarro.CARRO_DIST;

  public CanvasRuta(double input[], double angulos[], double velocidades[],
                    CambioCoordenadas cc, int ancho, int alto) {

    setLimites(input, ancho, alto);

    setRuta(input, angulos, velocidades);

    this.setBackground(Color.white);
    this.setSize(this.ancho, this.alto);
    this.cc = cc;
    addMouseListener(this);
    addKeyListener(this);
  }

  public void setLimites(double input[], int ancho, int alto) {
    minX = maxX = input[0];
    minY = maxY = input[1];

    for (int i = 2; i < input.length; i += 2) {
      if (input[i] > maxX)
        maxX = input[i];
      if (input[i] < minX)
        minX = input[i];
      if (input[i + 1] > maxY)
        maxY = input[i + 1];
      if (input[i + 1] < minY)
        minY = input[i + 1];
    }

    double distX = Math.abs(maxX - minX);
    double distY = Math.abs(maxY - minY);

    minX -= distX * 0.1;
    maxX += distX * 0.1;
    minY -= distY * 0.1;
    maxY += distY * 0.1;

    if (distX > distY) {
      this.ancho = ancho;
      this.alto = (int)(ancho * (distY / distX));
    } else {
      this.ancho = (int)(alto * (distX / distY));
      this.alto = alto;
    }

  }

  public void setRuta(double input[], double angulos[], double velocidades[]) {
    ruta = new Point[input.length / 2];

    kx = ancho / (maxX - minX);
    ky = alto / (maxY - minY);

    for (int i = 0; i < input.length; i += 2) {
      ruta[i / 2] = new Point();
      ruta[i / 2].x = (int)(kx * (input[i] - minX));//(int)((ancho * (input[i] - minX)) / (maxX - minX));
      ruta[i / 2].y = alto - (int)(ky * (input[i + 1] - minY));//alto - (int)((alto * (input[i + 1] - minY)) / (maxY - minY));
    }

    this.velocidades = velocidades;
    this.angulos = angulos;

    vectores = new Point[angulos.length];
    for (int i = 0; i < vectores.length; i++) {
      vectores[i] = new Point((int)(20 * Math.cos(angulos[i])), (int)(20 * Math.sin(-angulos[i])));
    }

  }

  public void setRuta2(double input2[], double angulos2[], double velocidades2[]) {
      ruta2 = new Point[input2.length / 2];

      for (int i = 0; i < input2.length; i += 2) {
        ruta2[i / 2] = new Point();
        ruta2[i / 2].x = (int)(kx * (input2[i] - minX));
        ruta2[i / 2].y = alto - (int)(ky * (input2[i + 1] - minY));
      }

      this.velocidades2 = velocidades2;
      this.angulos2 = angulos2;

      vectores2 = new Point[angulos2.length];
      for (int i = 0; i < vectores2.length; i++) {
        vectores2[i] = new Point((int)(20 * Math.cos(angulos2[i])), (int)(20 * Math.sin(-angulos2[i])));
      }

    }


  public void setPoint(double x, double y, double[] u, double angulo, double velocidad) {
    trayectoria.add(p.clone());

    //u[1] = u[1] + angulo % 360;
    int x2 = (int)(kx * (x - minX));
    int y2 = alto - (int)(ky * (y - minY));

    miAngulo = angulo;
    miVelocidad = velocidad;
    miVector.setLocation((int)(20 * Math.cos(angulo)) + x2, (int)(20 * Math.sin(-angulo)) + y2);

    giroRuedas = u[0];
    velEstimada = u[1];

    double ang = angulo + giroRuedas;
    if (ang < 0)
      ang += Math.toRadians(360);

    estima.setLocation((int)(20 * Math.cos(ang)) + x2, (int)(20 * Math.sin(-ang)) + y2);

    p.setLocation(x2,y2);

    repaint();
  }

  public void setCercano(int i) {
    //int x2 = (int)(kx * (x - minX));
    //int y2 = alto - (int)(ky * (y - minY));

    cercano = i / 2;
  }

  public void setAnguloRuedas(double anguloRuedas) {
    this.anguloRuedas = anguloRuedas;
  }

  public void addObstaculo(double input[]) {
    int obstaculox[] = new int[input.length / 2];
    int obstaculoy[] = new int[input.length / 2];

    for (int i = 0; i < input.length - 1; i += 2) {
      obstaculox[i / 2] = (int)(kx * (input[i] - minX));
      obstaculoy[i / 2] = alto - (int)(ky * (input[i + 1] - minY));
      System.out.println("(" + obstaculox[i / 2] + ", " + obstaculoy[i / 2] + ")");
      System.out.println("(" + input[i] + ", " + input[i + 1] + ")");
    }

    obstaculos.add(new Polygon(obstaculox, obstaculoy, obstaculox.length));
  }


  public void paint(Graphics g) {
    g.translate(origenX, origenY);

    if ((obstaculos != null) && (obstaculos.size() != 0)) {
      g.setColor(Color.darkGray);
      for (int i = 0; i < obstaculos.size(); i++) {
        Polygon pol = (Polygon) obstaculos.elementAt(i);
        int d[] = pol.xpoints;
        for (i = 0; i < d.length; i++) {
          d[i] *= aumento;
        }
        pol.xpoints = d;
        d = pol.ypoints;
        for (i = 0; i < d.length; i++) {
          d[i] *= aumento;
        }
        pol.ypoints = d;

        g.fillPolygon( pol );
      }
    }
    if ((ruta2 != null) && (ruta2.length != 0)) {
      g.setColor(Color.blue);
      for (int i = 1; i < ruta2.length; i++) {
        g.drawLine((int)(ruta2[i].x * aumento), (int)(ruta2[i].y * aumento),
                   (int)((vectores2[i].x + ruta2[i].x) * aumento), (int)((vectores2[i].y + ruta2[i].y) * aumento));
      }

      g.setColor(Color.gray);
      for (int i = 1; i < ruta2.length; i++) {
        g.drawLine((int)(ruta2[i - 1].x * aumento), (int)(ruta2[i - 1].y * aumento),
           (int)(ruta2[i].x * aumento), (int)(ruta2[i].y * aumento));
      }

      g.setColor(Color.black);
      for (int i = 0; i < ruta2.length; i++) {
        g.drawLine((int)((ruta2[i].x - 2) * aumento), (int)((ruta2[i].y - 2) * aumento),
                   (int)((ruta2[i].x + 2) * aumento), (int)((ruta2[i].y + 2) * aumento));
        g.drawLine((int)((ruta2[i].x - 2) * aumento), (int)((ruta2[i].y + 2) * aumento),
                   (int)((ruta2[i].x + 2) * aumento), (int)((ruta2[i].y - 2) * aumento));
      }

    }
    if ((ruta != null) && (ruta.length != 0)) {
      g.setColor(Color.blue);
      for (int i = 1; i < ruta.length; i++) {
        g.drawLine((int)(ruta[i].x * aumento), (int)(ruta[i].y * aumento),
                   (int)((vectores[i].x + ruta[i].x) * aumento),
                   (int)((vectores[i].y + ruta[i].y) * aumento));
      }

      g.setColor(Color.yellow);
      for (int i = 1; i < ruta.length; i++) {
        g.drawLine((int)(ruta[i - 1].x * aumento), (int)(ruta[i - 1].y * aumento),
                   (int)(ruta[i].x * aumento), (int)(ruta[i].y * aumento));
      }

      g.setColor(Color.black);
      for (int i = 0; i < ruta.length; i++) {
        g.drawLine((int)((ruta[i].x - 2) * aumento), (int)((ruta[i].y - 2) * aumento),
                   (int)((ruta[i].x + 2) * aumento), (int)((ruta[i].y + 2) * aumento));
        g.drawLine((int)((ruta[i].x - 2) * aumento), (int)((ruta[i].y + 2) * aumento),
                   (int)((ruta[i].x + 2) * aumento), (int)((ruta[i].y - 2) * aumento));
      }

    }
    g.setColor(Color.GRAY);
    for (int i = 1; i < trayectoria.size(); i++) {
      Point a = (Point)trayectoria.elementAt(i - 1);
      Point b = (Point)trayectoria.elementAt(i);
      g.drawLine((int)(a.x * aumento), (int)(a.y * aumento),
                 (int)(b.x * aumento), (int)(b.y * aumento));
    }
    Point p = new Point(this.p.x, this.p.y);
    p.setLocation((int)(p.x * aumento), (int)(p.y * aumento));
    Point miVector = new Point(this.miVector.x, this.miVector.y);;
    miVector.setLocation((int)(miVector.x * aumento), (int)(miVector.y * aumento));
    if (trayectoria.size() > 0) {
      Point a = (Point) trayectoria.lastElement();
      a = new Point((int)(a.x * aumento), (int)(a.y * aumento));
      g.drawLine(a.x, a.y, p.x, p.y);
      if (trayectoria.size() > 100) {
        trayectoria.remove(0);
      }
    }

    g.setColor(Color.red);
    g.drawLine(p.x, p.y, (int)(estima.x * aumento), (int)(estima.y * aumento));
    g.setColor(Color.BLUE);
    g.fillOval(p.x - 3, p.y - 3, 6, 6);
    g.drawLine(p.x, p.y, miVector.x, miVector.y);

    g.setColor(Color.green);
    g.fillOval((int)((ruta[cercano].x - 3) * aumento), (int)((ruta[cercano].y - 3) * aumento),
               (int)(6 * aumento), (int)(6 * aumento));
    g.drawLine((int)(ruta[cercano].x * aumento), (int)(ruta[cercano].y * aumento),
               (int)((ruta[cercano].x + vectores[cercano].x) * aumento), (int)((ruta[cercano].y + vectores[cercano].y) * aumento));

    /*// Simulacion
    g.setColor(Color.red);
    Point last = new Point(p.x, p.y);
    double lastGiro = miAngulo;
    for (int i = 0; i < 100; i++) {
      double miX = ((last.x - origenX) / (kx * aumento) ) + minX;
      double miY = ((alto * aumento - last.y + origenY) / (ky * aumento)) + minY;
      double u[] = cc.getU(miX, miY, lastGiro, miVelocidad);
      double giro = u[0];
      double velo = u[1];
      lastGiro += giro;
      Point next = new Point((int)((last.x + (simulaVel * Math.cos(lastGiro))) * aumento),
                 (int)((last.y + (simulaVel * Math.sin(-lastGiro))) * aumento));
      g.drawLine(last.x, last.y, next.x, next.y);
      last = next.getLocation();
    }
    //*/

    //Dibuja las ruedas del coche
    double angR = anguloRuedas  * Math.toRadians(100) / (ControlCarro.CARRO_DIST + ControlCarro.CARRO_CENTRO);
    angR += Math.toRadians(35);
    System.out.println(anguloRuedas + " = " + Math.toDegrees(angR));
    int hr = 25;
    double distAng = 0.4;
    Point p1 = new Point((int)(hr * Math.cos(angR + distAng)) + ancho - 350 - origenX,
                         (int)(hr * Math.sin(angR + distAng)) + alto - 160 - origenY);
    Point p2 = new Point((int)(hr * Math.cos(angR - distAng)) + ancho - 350 - origenX,
                         (int)(hr * Math.sin(angR - distAng)) + alto - 160 - origenY);
    Point p3 = new Point(ancho - 350 - origenX - (int)(hr * Math.cos(angR + distAng)),
                         alto - 160 - origenY - (int)(hr * Math.sin(angR + distAng)));
    Point p4 = new Point(ancho - 350 - origenX - (int)(hr * Math.cos(angR - distAng)),
                     alto - 160 - origenY - (int)(hr * Math.sin(angR - distAng)));

    g.setColor(Color.darkGray);
    g.fillPolygon(new int[] { p1.x, p2.x, p3.x, p4.x },
        new int[] { p1.y, p2.y, p3.y, p4.y }, 4);
    p1.x += 150;
    p2.x += 150;
    p3.x += 150;
    p4.x += 150;
    g.fillPolygon(new int[] { p1.x, p2.x, p3.x, p4.x },
                  new int[] { p1.y, p2.y, p3.y, p4.y }, 4);
    g.setColor(Color.GREEN);
    g.fillRect(ancho - 350 - origenX, alto - 165 - origenY, 150, 100);

    g.setColor(Color.black);
    // Pinta la escala
    for (int i = 0; i < 10; i++) {
      int miX = 20 + i * (int)Math.round(kx * aumento);
      if ((i % 2) == 0) {
        g.fillRect(miX - origenX, 20 - origenY, (int)Math.round(kx * aumento), 6);
      } else {
        g.drawRect(miX - origenX, 20 - origenY, (int)Math.round(kx * aumento), 5);
      }
    }
    g.drawString("0", 16 - origenX, 15 - origenY);
    g.drawString("10 m.", (int)(16 + kx * aumento * 10) - origenX, 15 - origenY);

    g.drawString("Zoom: " + (int)(aumento * 100) + "%", ancho - 100 - origenX, 15 - origenY);

    g.drawString("Velocidad actual: " + miVelocidad, ancho - 160 - origenX, alto - 170 + 15 - origenY);
    g.drawString("Velocidad deseada: " + velocidades[cercano], ancho - 160 - origenX, alto - 170 + 30 - origenY);
    String ang = Double.toString(miAngulo * 180 /Math.PI) + ".";
    ang = ang.substring(0, ang.indexOf(".") + 3);
    g.drawString("Ángulo actual: " + ang , ancho - 160 - origenX, alto - 170 + 50 - origenY);
    ang = Double.toString(angulos[cercano] * 180 /Math.PI) + ".";
    ang = ang.substring(0, ang.indexOf(".") + 3);
    g.drawString("Ángulo deseado: " + ang, ancho - 160 - origenX, alto - 170 + 65 - origenY);
    ang = Double.toString(Math.toDegrees(giroRuedas)) + ".";
    ang = ang.substring(0, ang.indexOf(".") + 3);
    g.drawString("Necesario girar (U1): " + ang, ancho - 160 - origenX, alto - 170 + 80 - origenY);
    g.drawString("Velocidad est. (U2): " + velEstimada, ancho - 160 - origenX, alto - 170 + 95 - origenY);
    g.drawString("Consigna: " + cercano, ancho - 160 - origenX, alto - 170 + 115 - origenY);
    g.drawString("K: " + cc.getKAngulo(), ancho - 160 - origenX, alto - 170 + 135 - origenY);
    String miVelo = Double.toString((5 * simulaVel / (kx * aumento)) / 0.2778) + ".";
    miVelo = miVelo.substring(0, miVelo.indexOf(".") + 3);
    g.drawString("Velocidad sim.: " + miVelo + " km/h", ancho - 160 - origenX, alto - 170 + 150 - origenY);
  }

  public double getMaxX() {
    return maxX;
  }

  public double getMaxY() {
    return maxY;
  }

  public double getMinX() {
    return minX;
  }

  public double getMinY() {
    return minY;
  }

  public CambioCoordenadas getCc() {
    return cc;
  }

  public void mouseClicked(MouseEvent e) {
    if (e.getButton() == MouseEvent.BUTTON1) {
      int min = 0;

      double minDist = Math.sqrt(Math.pow(e.getPoint().x - ruta[0].x, 2) +
          Math.pow(e.getPoint().y - ruta[0].y, 2));
      for (int i = 1; i < ruta.length; i++) {
        double distancia = Math.sqrt(Math.pow(e.getPoint().x - ruta[i].x, 2) +
          Math.pow(e.getPoint().y - ruta[i].y, 2));
        if (distancia < minDist) {
          minDist = distancia;
          min = i;
        }
      }
      //((CanvasRuta)e.getSource()).setCercano(min * 2);
      double miX = (((e.getPoint().x - origenX) / (kx * aumento) ) + minX);
      double miY = (((alto * aumento - e.getPoint().y + origenY) / (ky * aumento)) + minY);

      cc.muestraImagenCercana(miX, miY);
      cc.simulaPuntoActual(miX, miY, miAngulo + giroRuedas, miVelocidad);
      repaint();

    }
  }
  public void mouseEntered(MouseEvent e) {}
  public void mouseExited(MouseEvent e) {}
  public void mousePressed(MouseEvent e) {}
  public void mouseReleased(MouseEvent e) {}

  public void keyPressed(KeyEvent e) {
    if (e.getKeyCode() == KeyEvent.VK_PAGE_UP) {
      double last = aumento;
      aumento -= 0.1;
      origenX *=  aumento / last;
      origenY *=  aumento / last;
      repaint();
    }
    if (e.getKeyCode() == KeyEvent.VK_PAGE_DOWN) {
      double last = aumento;
      aumento += 0.1;
      origenX *= aumento / last;
      origenY *= aumento / last;
      repaint();
    }

    if (e.getKeyCode() == KeyEvent.VK_W) {
      origenY += 10;
      repaint();
    }

    if (e.getKeyCode() == KeyEvent.VK_S) {
      origenY -= 10;
      repaint();
    }

    if (e.getKeyCode() == KeyEvent.VK_D) {
      origenX -= 10;
      repaint();
    }

    if (e.getKeyCode() == KeyEvent.VK_A) {
      origenX += 10;
      repaint();
    }

    if (e.getKeyCode() == KeyEvent.VK_HOME) {
      origenX = origenY = 0;
      aumento = 1;
      repaint();
    }

    if (e.getKeyCode() == KeyEvent.VK_UP) {
      double angulo = ((2 * Math.PI) + (miAngulo + (Math.PI / 180 * 2))) % (2 * Math.PI);

      double miX = (p.x / kx) + minX;
      double miY = ((alto - p.y) / ky) + minY;
      cc.simulaPuntoActual(miX, miY, angulo, miVelocidad);
      repaint();

    }

    if (e.getKeyCode() == KeyEvent.VK_DOWN) {
      double angulo = (miAngulo - (Math.PI / 180 * 2)) % (2 * Math.PI);

      double miX = (p.x / kx) + minX;
      double miY = ((alto - p.y) / ky) + minY;
      cc.simulaPuntoActual(miX, miY, angulo, miVelocidad);
      repaint();

    }

    if (e.getKeyCode() == KeyEvent.VK_LEFT) {

      double velocidad = miVelocidad - 0.25;

      double miX = (p.x / kx) + minX;
      double miY = ((alto - p.y) / ky) + minY;
      cc.simulaPuntoActual(miX, miY, miAngulo, velocidad);
      repaint();

    }


    if (e.getKeyCode() == KeyEvent.VK_RIGHT) {

      double velocidad = miVelocidad + 0.25;

      double miX = (p.x / kx) + minX;
      double miY = ((alto - p.y) / ky) + minY;
      cc.simulaPuntoActual(miX, miY, miAngulo, velocidad);
      repaint();
    }

    if (e.getKeyCode() == KeyEvent.VK_K) {
      cc.setKAngulo(cc.getKAngulo() + 0.01);

      double miX = (p.x / kx) + minX;
      double miY = ((alto - p.y) / ky) + minY;
      cc.simulaPuntoActual(miX, miY, miAngulo, miVelocidad);
      repaint();
    }

    if (e.getKeyCode() == KeyEvent.VK_L) {
      cc.setKAngulo(cc.getKAngulo() - 0.01);

      double miX = (p.x / kx) + minX;
      double miY = ((alto - p.y) / ky) + minY;
      cc.simulaPuntoActual(miX, miY, miAngulo, miVelocidad);
      repaint();
    }

    if (e.getKeyCode() == KeyEvent.VK_V) {
      simulaVel += 1;

      double miX = (p.x / kx) + minX;
      double miY = ((alto - p.y) / ky) + minY;
      cc.simulaPuntoActual(miX, miY, miAngulo, miVelocidad);
      repaint();
    }

    if (e.getKeyCode() == KeyEvent.VK_B) {
      simulaVel -= 1;
      if (simulaVel < 1)
        simulaVel = 1;

      double miX = (p.x / kx) + minX;
      double miY = ((alto - p.y) / ky) + minY;
      cc.simulaPuntoActual(miX, miY, miAngulo, miVelocidad);
      repaint();
    }

  }
  public void keyReleased(KeyEvent e) {}
  public void keyTyped(KeyEvent e) {
  }

}
