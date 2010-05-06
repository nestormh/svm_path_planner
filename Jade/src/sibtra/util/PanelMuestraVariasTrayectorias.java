package sibtra.util;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Stroke;
import java.awt.geom.GeneralPath;
import java.awt.geom.Point2D;
import java.util.Vector;

import javax.swing.JCheckBox;

import sibtra.gps.Trayectoria;


/**
 * Panel que usa {@link PanelMapa} para mostrar varias trayectorias la y posición del coche.
 * Por ahora es estático y no admite añadir un punto, sólo cambiar toda la trayectoria.
 * @author alberto
 */
@SuppressWarnings("serial")
public class PanelMuestraVariasTrayectorias extends PanelMapa {
	
	public Stroke strokeLinea=new BasicStroke();
	public Stroke strokeGruesa=new BasicStroke(2.0f);
	public Stroke strokeMuyGruesa=new BasicStroke(4.0f);
	
	/** Tamaño en pixeles del aspa que marca cada punto */ 
	protected static final int tamCruz = 2;
	
	/** Longitud del vector que marca el rumbo en cada punto */
	private static final double tamRumbo = 50;
	/** Distancia a la que tiene que encontrarse el coche de una trayectoria para que se marque el punto más cercano */
	private static final double HUMBRAL_MAS_CERCANO = 2;

	protected class ParamTra {
		Trayectoria trayec=null;
		boolean mostrar=true;
		boolean puntos=false;
		boolean rumbo=false;
		boolean destacado=false;
		Color color=Color.YELLOW;
		Vector<Integer> macrados=null;
		
		ParamTra(Trayectoria tr) {
			trayec=tr;
		}
	}
	
	/** Vector de trayectorias representadas */
	protected Vector<ParamTra> trays=new Vector<ParamTra>();

	public static Color[] colores={Color.BLUE, Color.CYAN, Color.DARK_GRAY
//		, Color.GRAY //Demasiado oscuro
			,Color.GREEN, Color.LIGHT_GRAY, Color.MAGENTA, Color.ORANGE, Color.PINK
			,Color.RED
			, Color.YELLOW, Color.WHITE};

	protected int indiceColor=0;

	/** coordenadas de la posición del coche. Si es NaN el coche no se pinta */
	protected double posXCoche=Double.NaN;
	protected double posYCoche;
	/** orientación del coche */
	protected double orientacionCoche;
	
	
	
	/** Para marcar si se quiere seguir el coche cuando hay cambios de posición */
	protected JCheckBox jcbSeguirCoche;
	
	/** Para marcar si se quiere mostrar el coche */
	protected JCheckBox jcbMostrarCoche;
		
    /**
     * Constructor 
     */
	public PanelMuestraVariasTrayectorias() {
		super();

		JCheckBox jcba;
		

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
		//pintamos las trayectorias mostradas con sus caracteríasticas
		for(int ita=0; ita<trays.size(); ita++) {
			ParamTra ptAct=trays.get(ita);
			Trayectoria traAct=ptAct.trayec;
			if(traAct.getLargo()==0 || !ptAct.mostrar)
				continue; //no tenemos que mostrarla
			if(ptAct.destacado)
				g.setStroke(strokeMuyGruesa);
			else
				g.setStroke(strokeLinea);
			g.setColor(ptAct.color);
			if(!ptAct.puntos) {
				GeneralPath gptr=pathTrayectoria(traAct);
				if(gptr!=null) {
					g.draw(gptr);
				}
			} else {
				puntosTrayectoria(g,traAct);
			}
			//Marcamos puntos si se a asignado vector de índice
			if(ptAct.macrados!=null && ptAct.macrados.size()>0) {
				//pintamos los puntos que están dentro del recuadro como cuadrados
				g.setStroke(strokeGruesa);
				for(int ia=0; ia<ptAct.macrados.size(); ia++)
					if (ptAct.macrados.get(ia)<traAct.length()) {
						double pa[]={traAct.x[ptAct.macrados.get(ia)],traAct.y[ptAct.macrados.get(ia)]};
						if(pa[0]<=esqSI.getX() && pa[0]>=esqID.getX()
								&& pa[1]<=esqSI.getY() && pa[1]>=esqID.getY() ) {
							//esta dentro del recuadro
							Point2D px=point2Pixel(pa);
							int x=(int)px.getX(), y=(int)px.getY();
							g.drawRect(x-tamCruz, y-tamCruz, tamCruz*2, tamCruz*2);
						}
					}
			}

			if(ptAct.rumbo) {
				g.setStroke(strokeLinea);
//				g.setColor(Color.BLUE);
				//pintamos los puntos que están dentro del recuadro
				for(int i=0; i<traAct.length(); i++) {
					double pa[]={traAct.x[i],traAct.y[i]};
					if(pa[0]<=esqSI.getX() && pa[0]>=esqID.getX()
							&& pa[1]<=esqSI.getY() && pa[1]>=esqID.getY() ) {
						//esta dentro del recuadro
						Point2D px=point2Pixel(pa);
						int x=(int)px.getX(), y=(int)px.getY();
						int Dx=(int)(-tamRumbo*Math.sin(traAct.rumbo[i]));
						int Dy=(int)(-tamRumbo*Math.cos(traAct.rumbo[i]));
						g.drawLine(x, y
								, x+Dx, y+Dy);
					}
				}

			}
			//Si el coche está situado y no se va a mostrar marcamos el más cercano de la trayectoria
			if(!Double.isNaN(posXCoche) && !jcbMostrarCoche.isSelected()) {
				//pintamos mas grueso el más cercano al coche
				g.setStroke(strokeMuyGruesa);
				traAct.situaCoche(posXCoche, posYCoche);
				int i=traAct.indiceMasCercano();
				if( traAct.distanciaAlMasCercano()<HUMBRAL_MAS_CERCANO &&
						traAct.x[i]<=esqSI.getX() && traAct.x[i]>=esqID.getX()
						&& traAct.y[i]<=esqSI.getY() && traAct.y[i]>=esqID.getY() ) {
					//esta dentro del recuadro
					Point2D px=point2Pixel(traAct.x[i],traAct.y[i]);
					int x=(int)px.getX(), y=(int)px.getY();
					g.drawRect(x-(tamCruz+2), y-(tamCruz+2), (tamCruz+2), (tamCruz+2));
//					g.drawLine(x-tamCruz, y-tamCruz
//							, x+tamCruz, y+tamCruz);
//					g.drawLine(x-tamCruz, y+tamCruz
//							, x+tamCruz, y-tamCruz);
				}
			}
		}

		//Mostramos el coche según corresponda
		if(Double.isNaN(posXCoche)) {
			jcbMostrarCoche.setEnabled(false);
			jcbSeguirCoche.setEnabled(false);
		} else {
			jcbMostrarCoche.setEnabled(true);
			jcbSeguirCoche.setEnabled(true);
			if(jcbMostrarCoche.isSelected()){
				//Posición y orientación del coche
				g.setStroke(strokeGruesa);
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
	 * Pinta los puntos de la trayectoria como aspas
	 * @param g donde pintar
	 * @param tra Trayectoria a pintar
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

	/** Los límites que necesitamos son los de la ruta a representar.
	 * Sólo tenemos en cuenta las trayectorias que se están mostrando
	 */
	protected double[] limites() {
		//double axis[]=super.limites();
		//no tomamos en cuenta los limites del padre
		double axis[]={Double.MAX_VALUE,Double.MIN_VALUE,
				Double.MAX_VALUE,Double.MIN_VALUE};
		boolean algunoMostrado=false;
		for(ParamTra ptAct: trays)
			if(ptAct.mostrar){
				algunoMostrado=true;
				axis[0]=UtilCalculos.minimo(axis[0],ptAct.trayec.x);
				axis[1]=UtilCalculos.maximo(axis[1],ptAct.trayec.x);
				axis[2]=UtilCalculos.minimo(axis[2],ptAct.trayec.y);
				axis[3]=UtilCalculos.maximo(axis[3],ptAct.trayec.y);
			}
		if(!Double.isNaN(posXCoche)) {
			algunoMostrado=true;
			if(posXCoche<axis[0]) axis[0]=posXCoche;
			if(posXCoche>axis[1]) axis[1]=posXCoche;
			if(posYCoche<axis[2]) axis[2]=posYCoche;
			if(posYCoche>axis[3]) axis[3]=posYCoche;

		}
		if(!algunoMostrado)
			//si no hay ninguna mostrada ni hay coche, usamos los del padre
			axis=super.limites();
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
	
	/** @return Ídem que {@link #pathTrayectoria(Trayectoria, int, int)} usando todo el array.	 */
	protected GeneralPath pathTrayectoria(Trayectoria tr) {
		if(tr==null)
			return null;
		return pathTrayectoria(tr, 0, tr.length());
		
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

	/** Añade nueva trayectoria a representar 
	 * No actuliza el panel 
	 * @return indice de la trayectoria
	 */
	public int añadeTrayectoria(Trayectoria tra) {
		//asignamos el siguiente color de la lista
		return añadeTrayectoria(tra, colores[(indiceColor++)%colores.length]);
	}
	
	/** Añade nueva trayectoria a representar
	 * No actuliza el panel 
	 * @return indice de la trayectoria
	 */
	public int añadeTrayectoria(Trayectoria tra, Color color) {
		if(tra==null)
			throw new IllegalArgumentException("Trayectoria pasada no puede ser null");
		trays.add(new ParamTra(tra));
		//asignamos color indicado
		trays.lastElement().color=color;
		return trays.size()-1;
	}

	/** Cambia el parámetro correspondiente a la trayectoria i-ésima. No actualiza el panel */
	public void setMostrado(int i, boolean valor) {
		if(i<0 || i>=trays.size())
			throw new IllegalArgumentException("Indice de trayectoria ("+i+")fuera de rango");
		trays.get(i).mostrar=valor;
	}
	
	public boolean isMostrado(int i) {
		if(i<0 || i>=trays.size())
			throw new IllegalArgumentException("Indice de trayectoria ("+i+")fuera de rango");
		return trays.get(i).mostrar;
	}

	/** Cambia el parámetro correspondiente a la trayectoria i-ésima. No actualiza el panel */
	public void setPuntos(int i, boolean valor) {
		trays.get(i).puntos=valor;
	}

	public boolean isPuntos(int i){
		if(i<0 || i>=trays.size())
			throw new IllegalArgumentException("Indice de trayectoria ("+i+")fuera de rango");
		return trays.get(i).puntos;
	}
	/** Cambia el parámetro correspondiente a la trayectoria i-ésima. No actualiza el panel */
	public void setRumbo(int i, boolean valor) {
		if(i<0 || i>=trays.size())
			throw new IllegalArgumentException("Indice de trayectoria ("+i+")fuera de rango");
		trays.get(i).rumbo=valor;
	}

	public boolean isRumbo(int i){
		if(i<0 || i>=trays.size())
			throw new IllegalArgumentException("Indice de trayectoria ("+i+")fuera de rango");
		return trays.get(i).rumbo;
	}
	/** Cambia el parámetro correspondiente a la trayectoria i-ésima. No actualiza el panel */
	public void setDestacado(int i, boolean valor) {
		if(i<0 || i>=trays.size())
			throw new IllegalArgumentException("Indice de trayectoria ("+i+")fuera de rango");
		trays.get(i).destacado=valor;
	}

	public boolean isDestacado(int i){
		if(i<0 || i>=trays.size())
			throw new IllegalArgumentException("Indice de trayectoria ("+i+")fuera de rango");
		return trays.get(i).destacado;
	}
	/** Cambia el parámetro correspondiente a la trayectoria i-ésima. No actualiza el panel */
	public void setColor(int i, Color col) {
		if(i<0 || i>=trays.size())
			throw new IllegalArgumentException("Indice de trayectoria ("+i+")fuera de rango");
		trays.get(i).color=col;
	}
	
	
	public Color getColor(int i) {
		if(i<0 || i>=trays.size())
			throw new IllegalArgumentException("Indice de trayectoria ("+i+")fuera de rango");
		return trays.get(i).color;		
	}
	
	/** Cambia el parámetro correspondiente a la trayectoria i-ésima. No actualiza el panel */
	public void setTrayectoria(int i, Trayectoria tra) {
		if(i<0 || i>=trays.size())
			throw new IllegalArgumentException("Indice de trayectoria ("+i+")fuera de rango");
		if(tra==null)
			throw new IllegalArgumentException("Trayectoria pasada no puede ser null");
		trays.get(i).trayec=tra;
	}

	/** @return la trayectoria i-ésima. */
	public Trayectoria getTrayectoria(int i) {
		if(i<0 || i>=trays.size())
			throw new IllegalArgumentException("Indice de trayectoria ("+i+")fuera de rango");
		return trays.get(i).trayec;
	}

	/** Cambia el parámetro correspondiente a la trayectoria i-ésima. No actualiza el panel */
	public void setMarcados(int i, Vector<Integer> mar) {
		if(i<0 || i>=trays.size())
			throw new IllegalArgumentException("Indice de trayectoria ("+i+")fuera de rango");
		trays.get(i).macrados=mar;
	}

	/** Cambia el parámetro correspondiente a la trayectoria i-ésima. No actualiza el panel */
	public void borraTrayectoria(int i) {
		if(i<0 || i>=trays.size())
			throw new IllegalArgumentException("Indice de trayectoria ("+i+")fuera de rango");
		trays.remove(i);
	}

	public void setSeguirCoche(boolean val) {
		jcbSeguirCoche.setSelected(val);
	}

	public boolean isSeguirCoche() {
		return jcbSeguirCoche.isSelected();
	}
	
	public void setMostrarCoche(boolean val) {
		jcbMostrarCoche.setSelected(val);
	}
	
	public boolean isMostrarCoche() {
		return jcbMostrarCoche.isSelected();
	}

	/** Hace que el tramo indicado pase a una posición anterior */
	public void subirTrayectoria(int i) {
		if(i<0 || i>=trays.size())
			throw new IllegalArgumentException("Índice fuera de rango");
		if(i==0)
			return; //no hacemos nada
		ParamTra dta=trays.get(i);
		trays.set(i,trays.get(i-1));
		trays.set(i-1, dta);
	}
	
	/** Hace que el tramo indicado pase a una posición posterior */
	public void bajarTrayectoria(int i) {
		if(i<0 || i>=trays.size())
			throw new IllegalArgumentException("Índice fuera de rango");
		if(i==(trays.size()-1))
			return; //no hacemos nada
		ParamTra dta=trays.get(i);
		trays.set(i,trays.get(i+1));
		trays.set(i+1, dta);
	}
	

}
