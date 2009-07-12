package sibtra.util;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.geom.GeneralPath;
import java.awt.geom.Point2D;
import java.util.Vector;

import javax.swing.JCheckBox;

import sibtra.gps.Trayectoria;


/**
 * Panel que usa {@link PanelMapa} para mostrar una trayectoria la y posición del coche.
 * Por ahora es estático y no admite añadir un punto, sólo cambiar toda la trayectoria.
 * @author alberto
 */
@SuppressWarnings("serial")
public class PanelMuestraTrayectoria extends PanelMapa {
	
	/** Tamaño en pixeles del aspa que marca cada punto */ 
	protected static final int tamCruz = 2;
	
	/** Longitud del vector que marca el rumbo en cada punto */
	private static final double tamRumbo = 50;

	/** Array de dos o tres columnas con los puntos que forman la trayectoria */
	protected Trayectoria tray=null;

	/** Vector de puntos de la trayectoria a marcar de manera especial */
	Vector<Integer> indiceMarcar=null;

	/** coordenadas de la posición del coche. Si es NaN el coche no se pinta */
	protected double posXCoche=Double.NaN;
	protected double posYCoche;
	/** orientación del coche */
	protected double orientacionCoche;
	
	
	
	/** Para marcar si se quiere seguir el coche cuando hay cambios de posición */
	protected JCheckBox jcbSeguirCoche;
	
	/** Para marcar si se quiere mostrar el coche */
	protected JCheckBox jcbMostrarCoche;
	
	/** Para marcar si se quiere mostrar los puntos */
	protected JCheckBox jcbMostrarPuntos;

	/** Para marcar si se quiere mostrar el rumbo */
	protected JCheckBox jcbMostrarRumbo;
	
    /**
     * Constructor 
     * @param rupas Ruta pasada.
     */
	public PanelMuestraTrayectoria() {
		super();

		JCheckBox jcba;
		
		jcbMostrarPuntos=jcba=new JCheckBox("Puntos");
		jpSur.add(jcba);
		jcba.setEnabled(false);
		jcba.addActionListener(this);
		jcba.setSelected(false);

		jcbMostrarRumbo=jcba=new JCheckBox("Rumbo");
		jpSur.add(jcba);
		jcba.setEnabled(false);
		jcba.addActionListener(this);
		jcba.setSelected(false);

		jcbSeguirCoche=jcba=new JCheckBox("Seguir Coche");
		jpSur.add(jcba);
		jcba.setEnabled(false);
		jcba.addActionListener(this);
		jcba.setSelected(true);
		
		jcbMostrarCoche=jcba=new JCheckBox("Mostrar Coche");
		jpSur.add(jcba);
		jcba.setEnabled(false);
		jcba.addActionListener(this);
		jcba.setSelected(true);

		
		
		//Si queremoa añadir algo al panel inferiro	
		//		jpSur.add(jcbEscalas);

	}

	protected void cosasAPintar(Graphics g0) {
		super.cosasAPintar(g0);
		Graphics2D g=(Graphics2D)g0;
		//pintamos el trayecto si lo hay
		if(tray!=null && tray.length()>0) {
			g.setStroke(new BasicStroke());
			g.setColor(Color.YELLOW);
			if(!jcbMostrarPuntos.isSelected()) {
				GeneralPath gptr=pathTrayectoria(tray);
				if(gptr!=null) {
					g.draw(gptr);
				}
			} else {
				puntosTrayectoria(g,tray);
			}
			//Marcamos puntos si se a asignado vector de índice
			if(indiceMarcar!=null && indiceMarcar.size()>0) {
				//pintamos los puntos que están dentro del recuadro
				g.setStroke(new BasicStroke());
				g.setColor(Color.RED);
				for(int ia=0; ia<indiceMarcar.size(); ia++)
					if (indiceMarcar.get(ia)<tray.length()) {
						double pa[]={tray.x[indiceMarcar.get(ia)],tray.y[indiceMarcar.get(ia)]};
						if(pa[0]<=esqSI.getX() && pa[0]>=esqID.getX()
								&& pa[1]<=esqSI.getY() && pa[1]>=esqID.getY() ) {
							//esta dentro del recuadro
							Point2D px=point2Pixel(pa);
							int x=(int)px.getX(), y=(int)px.getY();
							g.drawLine(x-tamCruz, y-tamCruz
									, x+tamCruz, y+tamCruz);
							g.drawLine(x-tamCruz, y+tamCruz
									, x+tamCruz, y-tamCruz);
						}
					}
			}

			if(jcbMostrarRumbo.isSelected()) {
				g.setStroke(new BasicStroke());
				g.setColor(Color.BLUE);
				//pintamos los puntos que están dentro del recuadro
				for(int i=0; i<tray.length(); i++) {
					double pa[]={tray.x[i],tray.y[i]};
					if(pa[0]<=esqSI.getX() && pa[0]>=esqID.getX()
							&& pa[1]<=esqSI.getY() && pa[1]>=esqID.getY() ) {
						//esta dentro del recuadro
						Point2D px=point2Pixel(pa);
						int x=(int)px.getX(), y=(int)px.getY();
						int Dx=(int)(-tamRumbo*Math.sin(tray.rumbo[i]));
						int Dy=(int)(-tamRumbo*Math.cos(tray.rumbo[i]));
						g.drawLine(x, y
								, x+Dx, y+Dy);
					}
				}

			}
		}

		if(Double.isNaN(posXCoche)) {
			jcbMostrarCoche.setEnabled(false);
			jcbSeguirCoche.setEnabled(false);
		} else {
			jcbMostrarCoche.setEnabled(true);
			jcbSeguirCoche.setEnabled(true);
			if(jcbMostrarCoche.isSelected()){
				//Posición y orientación del coche
				g.setStroke(new BasicStroke(3));
				g.setPaint(Color.GRAY);
				g.setColor(Color.GRAY);
				double[] esqDD={posXCoche+Parametros.anchoCoche/2*Math.sin(orientacionCoche)
						,posYCoche-Parametros.anchoCoche/2*Math.cos(orientacionCoche) };
				double[] esqDI={posXCoche-Parametros.anchoCoche/2*Math.sin(orientacionCoche)
						,posYCoche+Parametros.anchoCoche/2*Math.cos(orientacionCoche) };
				double[] esqPD={esqDD[0]-Parametros.largoCoche*Math.cos(orientacionCoche)
						,esqDD[1]-Parametros.largoCoche*Math.sin(orientacionCoche) };
				double[] esqPI={esqDI[0]-Parametros.largoCoche*Math.cos(orientacionCoche)
						,esqDI[1]-Parametros.largoCoche*Math.sin(orientacionCoche) };
				Point2D pxDD=point2Pixel(esqDD);
				Point2D pxDI=point2Pixel(esqDI);
				Point2D pxPD=point2Pixel(esqPD);
				Point2D pxPI=point2Pixel(esqPI);
				GeneralPath coche=new GeneralPath();
				coche.moveTo((float)pxDD.getX(),(float)pxDD.getY());
				coche.lineTo((float)pxPD.getX(),(float)pxPD.getY());
				coche.lineTo((float)pxPI.getX(),(float)pxPI.getY());
				coche.lineTo((float)pxDI.getX(),(float)pxDI.getY());
				coche.closePath();
				g.fill(coche);
				g.draw(coche);
			}
		}
	}

	/**
	 * Pinta los puntos del array como aspas
	 * @param g donde pintar
	 * @param v puntos a pintar (x,y)
	 */
	protected void puntosArray(Graphics2D g, double[][] v) {
		//pintamos los puntos que están dentro del recuadro
		for(int i=0; i<v.length; i++) {
			double pa[]=v[i];
			if(pa[0]<=esqSI.getX() && pa[0]>=esqID.getX()
					&& pa[1]<=esqSI.getY() && pa[1]>=esqID.getY() ) {
				//esta dentro del recuadro
				Point2D px=point2Pixel(pa);
				int x=(int)px.getX(), y=(int)px.getY();
				g.drawLine(x-tamCruz, y-tamCruz
						, x+tamCruz, y+tamCruz);
				g.drawLine(x-tamCruz, y+tamCruz
						, x+tamCruz, y-tamCruz);
			}
		}
	}

	/**
	 * Pinta los puntos del array como aspas
	 * @param g donde pintar
	 * @param v puntos a pintar (x,y)
	 */
	protected void puntosTrayectoria(Graphics2D g, Trayectoria tra) {
		//pintamos los puntos que están dentro del recuadro
		for(int i=0; i<tra.length(); i++) {
			if(tra.x[i]<=esqSI.getX() && tra.x[i]>=esqID.getX()
					&& tra.y[i]<=esqSI.getY() && tra.y[i]>=esqID.getY() ) {
				//esta dentro del recuadro
				Point2D px=point2Pixel(tra.x[i],tra.y[i]);
				int x=(int)px.getX(), y=(int)px.getY();
				g.drawLine(x-tamCruz, y-tamCruz
						, x+tamCruz, y+tamCruz);
				g.drawLine(x-tamCruz, y+tamCruz
						, x+tamCruz, y-tamCruz);
			}
		}
	}

	/** Los límites que necesitamos son los de la ruta a representar */
	protected double[] limites() {
		double axis[]=super.limites();
		if(tray!=null) {
			axis[0]=UtilCalculos.minimo(axis[0],tray.x);
			axis[1]=UtilCalculos.maximo(axis[1],tray.x);
			axis[2]=UtilCalculos.minimo(axis[2],tray.y);
			axis[3]=UtilCalculos.maximo(axis[3],tray.y);
		}
		if(!Double.isNaN(posXCoche)) {
			if(posXCoche<axis[0]) axis[0]=posXCoche;
			if(posXCoche>axis[1]) axis[1]=posXCoche;
			if(posYCoche<axis[2]) axis[2]=posYCoche;
			if(posYCoche>axis[3]) axis[3]=posYCoche;

		}
		return axis;
	}

	
	/**
	 * Genera {@link GeneralPath} con puntos en array
	 * @param v array de al menos 2 columnas. La primera se considera coordenada X, la segunda la Y
	 * @param iini indice del primer punto
	 * @param ifin indice siguiente del último punto
	 * @param esCerrado si debemos considerrar array circular
	 * @return {@link GeneralPath} con los puntos considerados
	 */
	protected GeneralPath pathArrayXY(double [][] v, int iini, int ifin, boolean esCerrado) {
		if(v==null || v.length==0 || v[0].length<2 || iini<0 || ifin>v.length
				|| (!esCerrado && ifin<=iini)
				)
			return null;
		int numPuntos=(iini<ifin)?(ifin-iini):(v.length-iini+ifin);
		GeneralPath perimetro = 
			new GeneralPath(GeneralPath.WIND_EVEN_ODD, numPuntos);

		Point2D.Double px=point2Pixel(v[iini][0],v[iini][1]);
		perimetro.moveTo((float)px.getX(),(float)px.getY());
		int i=(iini+1)%v.length;
		for(int cont=2; cont<=numPuntos; cont++) {
			px=point2Pixel(v[i][0],v[i][1]);
			//Siguientes puntos son lineas
			perimetro.lineTo((float)px.getX(),(float)px.getY());
			i=(i+1)%v.length;
		}
		return perimetro;
	}

	/** Ídem {@link #pathArrayXY(double[][], int, int, boolean)} con esCerrado=false */
	protected GeneralPath pathArrayXY(double [][] v, int iini, int ifin) {
		return pathArrayXY(v, iini, ifin, false);
	}

	
	/** @return Ídem que {@link #pathArrayXY(double[][], int, int, boolean)} usando todo el array.	 */
	protected GeneralPath pathArrayXY(double[][] v, boolean esCerrada) {
		if(v==null)
			return null;
		return pathArrayXY(v, 0, v.length,esCerrada);
		
	}
	
	/** Ídem que {@link #pathArrayXY(double[][], boolean)} con cerrada =false */
	protected GeneralPath pathArrayXY(double[][] v) {
		return pathArrayXY(v, false);
	}
	
	/**
	 * Genera {@link GeneralPath} con cierto rango de puntos en {@link Trayectoria}. 
	 * Si la trayectoria es cerrada, indice final puede ser menor que el inicial.
	 * @param tr {@link Trayectoria} de la que sacar los puntos
	 * @param iini indice del primer punto
	 * @param ifin indice siguiente del último punto
	 * @return {@link GeneralPath} con los puntos considerados
	 */
	protected GeneralPath pathTrayectoria(Trayectoria tr, int iini, int ifin) {
		if(tr==null || tr.length()==0 
				|| iini<0 || iini>tr.length() 
				|| ifin<0 || ifin>tr.length()
				|| (!tr.esCerrada() && ifin<=iini)
				)
			return null;
		int numPuntos=(iini<ifin)?(ifin-iini):(tr.length()-iini+ifin);
		GeneralPath perimetro = 
			new GeneralPath(GeneralPath.WIND_EVEN_ODD, numPuntos);

		Point2D.Double px=point2Pixel(tr.x[iini],tr.y[iini]);
		perimetro.moveTo((float)px.getX(),(float)px.getY());
		int i=(iini+1)%tr.length();
		for(int cont=2; cont<=numPuntos; cont++) {
			px=point2Pixel(tr.x[i],tr.y[i]);
			//Siguientes puntos son lineas
			perimetro.lineTo((float)px.getX(),(float)px.getY());
			i=(i+1)%tr.length();
		}
		return perimetro;
	}
	
	/** @return Ídem que {@link #pathTrayectoria(Trayectoria, int, int, boolean)} usando todo el array.	 */
	protected GeneralPath pathTrayectoria(Trayectoria tr) {
		if(tr==null)
			return null;
		return pathTrayectoria(tr, 0, tr.length());
		
	}
	

	/** Establece la trayectoria a representar, pero no actualiza el panel
	 * 
	 * @param tr debe tener al menos 2 columnas
	 */
	public void setTr(Trayectoria tr) {
		if(tr==null || tr.length()==0) {
			jcbMostrarPuntos.setEnabled(false);
			jcbMostrarRumbo.setEnabled(false);
		} else {
			jcbMostrarPuntos.setEnabled(true);
			jcbMostrarRumbo.setEnabled(true);
		}
		tray=tr;
	}
	
	/** @param im vector de indice de puntos a marcar, null para no marcar */
	public void setMarcados(Vector<Integer> im) {
		indiceMarcar=im;
	}

	
	/** Define posición y orientación del coche. No repinta (usar {@link #actualiza()})
	 * @param posX si se pasa NaN el coche no se pinta (no está situado)
	 */
	public void situaCoche(double posX, double posY, double orientacion) {
		posXCoche=posX;
		posYCoche=posY;
		orientacionCoche=orientacion;
		if(!Double.isNaN(posXCoche) && jcbSeguirCoche.isSelected())
			setCentro(posXCoche,posYCoche);
	}

//	/**
//	 * Actualiza la presentación cuando la ruta tiene un nuevo punto. 
//	 */
//	public void nuevoPunto() {
//		GPSData ultPto=RU.getUltimoPto();
//		double x=ultPto.getXLocal();
//		double y=ultPto.getYLocal();
//		double yaw=ultPto.getAngulo();
//		if(jcbSeguirCoche.isSelected())
//			fijarCentro(x,y);
//		if(ultPto.getAngulosIMU()!=null)
//			yaw=ultPto.getAngulosIMU().getYaw();
//		situaCoche(x, y, yaw);
//		actualiza();
//	}
	


}
