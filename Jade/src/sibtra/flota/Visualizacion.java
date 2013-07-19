package sibtra.flota;

import javax.swing.*;
import java.awt.*;
import java.io.*;
import java.awt.event.*;
import java.util.*;
import javax.imageio.*;
import java.awt.image.*;

public class Visualizacion extends JPanel
{
String prefijo="http://www.isaatc.ull.es/Verdino.owl#";
String[] tramos;	
double[] longitudes;
Hashtable hashTramos = new Hashtable(); 
Hashtable hashColores = new Hashtable(); 
Hashtable hashVehiculos = new Hashtable(); 

private static final GridBagConstraints gbc;
static {
gbc = new GridBagConstraints();
gbc.gridx = 0;
gbc.gridy = 0;
gbc.weightx = 1;
gbc.weighty = 1;
gbc.fill = GridBagConstraints.BOTH;
gbc.anchor = GridBagConstraints.NORTHWEST;


}

String buffered = "lib/flota/12071394445.jpg";
BufferedImage image; 
public Visualizacion () throws IOException
{JFrame frame = new JFrame();
 frame.setTitle("VERDINO: Gesti�n de flota");
 image = ImageIO.read(new File(buffered));
 frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
// java.net.URL imgURL = getClass().getResource("12071394445.jpg");
// ImageIcon icon = new ImageIcon(imgURL, "Fondo");
  ImageIcon icon = new ImageIcon(buffered, "Fondo");
 JLabel label = new JLabel(icon);
  JPanel backgroundPanel = new JPanel(new GridBagLayout());
 backgroundPanel.add(label,gbc);
 this.setOpaque(false);
 backgroundPanel.add(this, gbc);
  backgroundPanel.setSize(750,750);
 frame.setContentPane(this);
 frame.setSize(740, 770);
 frame.setVisible(true);
  
 double[] longitudes = {1, 260, 200, 290, 490, 290, 250, 290, 200, 250, 725, 402, 402, 350, 350, 150, 150, 390, 390, 350, 350, 200, 20, 20, 30};
tramos = new String[longitudes.length];
	 for (int j= 0; j< tramos.length; j++)
	 {tramos[j] = prefijo + "Tramo" + (j);
	 }	
 TramoPintable t0 = new TramoPintable(0,0,0,0, longitudes[0]);
 TramoPintable t1 = new TramoPintable(90,10, 100,300, longitudes[1]);
 TramoPintable t2 = new TramoPintable(100,300,110,430, longitudes[2]);
 TramoPintable t3 = new TramoPintable(120,350,105,10, longitudes[3]);
 TramoPintable t4 = new TramoPintable(120,350,470,350, longitudes[4]);
 TramoPintable t5 = new TramoPintable(470,350,470,110, longitudes[5]);
 TramoPintable t6 = new TramoPintable(470,110,580,110, longitudes[6]);
 TramoPintable t7 = new TramoPintable(580,110,580,350, longitudes[7]);
 TramoPintable t8 = new TramoPintable(580,350,580,430, longitudes[8]);
 TramoPintable t9 = new TramoPintable(470,350,580,350, longitudes[9]);
 TramoPintable t10 = new TramoPintable(580,430,180,430, longitudes[10]);
 TramoPintable t11 = new TramoPintable(250,510,180,430, longitudes[11]);
 TramoPintable t12 = new TramoPintable(180,430,250,510, longitudes[12]);
 TramoPintable t13 = new TramoPintable(250,510,480,600, longitudes[13]);
 TramoPintable t14 = new TramoPintable(480,600,250,510, longitudes[14]);
 TramoPintable t15 = new TramoPintable(250,510,400,725, longitudes[15]);
 TramoPintable t16 = new TramoPintable(400,725,250,510, longitudes[16]);
 TramoPintable t17 = new TramoPintable(400,725,10,725, longitudes[17]);
 TramoPintable t18 = new TramoPintable(10,725,400,725, longitudes[18]);
 TramoPintable t19 = new TramoPintable(400,725,750,725, longitudes[19]);
 TramoPintable t20 = new TramoPintable(700,725,400,725, longitudes[20]);
 TramoPintable t21 = new TramoPintable(120,430,120,350, longitudes[21]); 
 TramoPintable t22 = new TramoPintable(105,350,120,350, longitudes[22]); 
 TramoPintable t23 = new TramoPintable(110,430,180,430, longitudes[23]); 
 TramoPintable t24 = new TramoPintable(100,300,105,350, longitudes[24]); 
 
 hashTramos.put(tramos[0],t0);
 hashTramos.put(tramos[1],t1);
 hashTramos.put(tramos[2],t2);
 hashTramos.put(tramos[3],t3);
 hashTramos.put(tramos[4],t4);
 hashTramos.put(tramos[5],t5);
 hashTramos.put(tramos[6],t6);
 hashTramos.put(tramos[7],t7);
 hashTramos.put(tramos[8],t8);
 hashTramos.put(tramos[9],t9);
 hashTramos.put(tramos[10],t10);
 hashTramos.put(tramos[11],t11);
 hashTramos.put(tramos[12],t12);
 hashTramos.put(tramos[13],t13);
 hashTramos.put(tramos[14],t14);
 hashTramos.put(tramos[15],t15);
 hashTramos.put(tramos[16],t16);
 hashTramos.put(tramos[17],t17);
 hashTramos.put(tramos[18],t18);
 hashTramos.put(tramos[19],t19);
 hashTramos.put(tramos[20],t20);
 hashTramos.put(tramos[21],t21);
 hashTramos.put(tramos[22],t22);
  hashTramos.put(tramos[23],t23);
  hashTramos.put(tramos[24],t24);

  hashColores.put(prefijo + "Verdino", (Color.red));

 hashColores.put(prefijo + "Verdino2", (Color.green));
 hashColores.put(prefijo + "Verdino3", (Color.cyan));
  hashColores.put(prefijo + "Verdino4", (Color.black));
 
 VehiculoPintable v0 = new VehiculoPintable(600,400);
 VehiculoPintable v1 = new VehiculoPintable(600,500);
 VehiculoPintable v2 = new VehiculoPintable(600,600);
 VehiculoPintable v3 = new VehiculoPintable(600,650);
 hashVehiculos.put(prefijo + "Verdino", v0);
 hashVehiculos.put(prefijo + "Verdino2", v1);
 hashVehiculos.put(prefijo + "Verdino3", v2);
hashVehiculos.put(prefijo + "Verdino4", v3);	 
}

public void paintComponent(Graphics g)
{super.paintComponent(g);
 Graphics2D g2d =(Graphics2D) g;
 super.paintComponent(g2d);
 g.drawImage(image, 0,0, this);
 g2d.setColor(Color.white);
 for (Enumeration e = hashTramos.elements() ; e.hasMoreElements() ;) 
	{TramoPintable p = (TramoPintable) e.nextElement();
	 g2d.drawLine(p.dimex0(),p.dimey0(), p.dimex1(), p.dimey1());
    }
for (Enumeration e = hashVehiculos.keys() ; e.hasMoreElements() ;) 
	{String key = (String) e.nextElement();
	 VehiculoPintable p = (VehiculoPintable) (hashVehiculos.get(key));
	g2d.setColor((Color) (hashColores.get(key)));
	g2d.fillRect(p.dimex0() -10 ,p.dimey0() -10, 20, 20);
     }	 
}

public void recibeInformacion(String id, String tramo, double longitud)
{TramoPintable tp = (TramoPintable) hashTramos.get(tramo);
 int xtp0 = tp.dimex0();
 int ytp0 = tp.dimey0();
 int xtp1 = tp.dimex1();
 int ytp1 = tp.dimey1();
 double longtp = tp.dimeLongitud();
 VehiculoPintable vp = (VehiculoPintable) hashVehiculos.get(id);
 float xvp = (float) ((xtp1 - xtp0)*longitud/longtp + xtp0);
  float yvp = (float) ((ytp1 - ytp0)*longitud/longtp + ytp0);
 vp.fijax0(Math.round(xvp));
 vp.fijay0(Math.round(yvp));
 repaint();
}

public void fijaTramos(String[] tramos)
{this.tramos = tramos;
}

public void fijaLongitudes(double[] longitudes)
{this.longitudes = longitudes;
}

public void fijaVehiculos(Hashtable hashVehiculos)
{
}
 } 