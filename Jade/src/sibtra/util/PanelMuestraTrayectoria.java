package sibtra.util;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.geom.GeneralPath;
import java.awt.geom.Point2D;
import java.util.Vector;

import javax.swing.JCheckBox;


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
	protected double Tr[][]=null;

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
		if(Tr!=null && Tr.length>0) {
			g.setStroke(new BasicStroke());
			g.setColor(Color.YELLOW);
			if(!jcbMostrarPuntos.isSelected()) {
				GeneralPath gptr=pathArrayXY(Tr);
				if(gptr!=null) {
					g.draw(gptr);
				}
			} else {
				puntosArray(g,Tr);
			}
			//Marcamos puntos si se a asignado vector de índice
			if(indiceMarcar!=null && indiceMarcar.size()>0) {
				//pintamos los puntos que están dentro del recuadro
				g.setStroke(new BasicStroke());
				g.setColor(Color.RED);
				for(int ia=0; ia<indiceMarcar.size(); ia++)
					if (indiceMarcar.get(ia)<Tr.length) {
						double pa[]=Tr[indiceMarcar.get(ia)];
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

			if(jcbMostrarRumbo.isSelected() && Tr[0].length>=3) {
				g.setStroke(new BasicStroke());
				g.setColor(Color.BLUE);
				//pintamos los puntos que están dentro del recuadro
				for(int i=0; i<Tr.length; i++) {
					double pa[]=Tr[i];
					if(pa[0]<=esqSI.getX() && pa[0]>=esqID.getX()
							&& pa[1]<=esqSI.getY() && pa[1]>=esqID.getY() ) {
						//esta dentro del recuadro
						Point2D px=point2Pixel(pa);
						int x=(int)px.getX(), y=(int)px.getY();
						int Dx=(int)(-tamRumbo*Math.sin(pa[2]));
						int Dy=(int)(-tamRumbo*Math.cos(pa[2]));
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

	/** Los límites que necesitamos son los de la ruta a representar */
	protected double[] limites() {
		double axis[]=super.limites();
		if(Tr!=null) {
			axis[0]=min(0,axis[0],Tr);
			axis[1]=max(0,axis[1],Tr);
			axis[2]=min(1,axis[2],Tr);
			axis[3]=max(1,axis[3],Tr);
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
		int numPuntos=(iini<ifin)?(ifin-iini):(Tr.length-iini+ifin);
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
	
	
	/** Establece la trayectoria a representar, pero no actualiza el panel
	 * 
	 * @param tr debe tener al menos 2 columnas
	 */
	public void setTr(double[][] tr) {
		if(tr==null || tr.length==0) {
			jcbMostrarPuntos.setEnabled(false);
			jcbMostrarRumbo.setEnabled(false);
		} else
			if (tr[0].length<2)
				throw new IllegalArgumentException("La trayectoria pasada no tiene 2 columnas");
			else {
				jcbMostrarPuntos.setEnabled(true);
				if(tr[0].length>=3)
					jcbMostrarRumbo.setEnabled(true);
			}
		Tr=tr;
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
