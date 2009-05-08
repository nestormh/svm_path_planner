/*
 * Creado el 22/02/2008
 *
 * Creado por Alberto Hamilcon con Eclipse
 */
package sibtra.rfyruta;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.geom.GeneralPath;
import java.awt.geom.Line2D;
import java.awt.geom.Point2D;

import javax.swing.Box;
import javax.swing.JFrame;
import javax.swing.JLabel;

import sibtra.lms.PanelMuestraBarrido;
import sibtra.lms.BarridoAngular;
import sibtra.lms.ZonaLMS;
import sibtra.lms.ZonaRadialLMS;
import sibtra.lms.ZonaRectangularLMS;
import sibtra.lms.ZonaSegmentadaLMS;

@SuppressWarnings("serial")
public class PanelMiraObstaculoSubjetivo extends PanelMuestraBarrido {

	private MiraObstaculo MI;

	private JLabel jlDistLin;

	private JLabel jlDistCamino;

    /**
     * Dado punto del mundo real lo pasa pixeles el coordenadas del RF.
     * @param ds
     * @return 
     */
    private Point2D.Double pointReal2pixel(double[] ds) {
		return pointReal2pixel(ds[0],ds[1]);
	}

    /**
     * Dado punto del mundo real lo pasa pixeles el coordenadas del RF.
     * @param xR coordenada x del mundo real
     * @param yR coordenada y del mundo real
     * @return punto en pixeles
     */
    private Point2D.Double pointReal2pixel(double xR,double yR) {
    	double Xrel=(xR-MI.posActual[0]);
    	double Yrel=(yR-MI.posActual[1]);
    	double angRot=MI.Yaw-Math.PI/2;
		return point2Pixel(new Point2D.Double(
				Xrel*Math.cos(angRot)+Yrel*Math.sin(angRot)
				,-Xrel*Math.sin(angRot)+Yrel*Math.cos(angRot)
				));
	}

    /**
	 * Obtiene posición relativa al RF de una medida del RF
	 * @param i indice del barrido a considerar
	 * @return posición real obtenidad a partir de posición actual y rumbo en {@link #MI}
	 */
	protected Point2D ptoRF2Point(int i) {
		double ang=MI.barr.getAngulo(i);
		double dist=MI.barr.getDistancia(i);
		return new Point2D.Double(dist*Math.cos(ang),dist*Math.sin(ang));
	}


	/**
	 * Crea parte grafica junto con slider de zoom
	 * @param distanciaMaxima Distancia máxima del gráfico
	 */
	public PanelMiraObstaculoSubjetivo(MiraObstaculo miObs,short distanciaMaxima) {
		super(distanciaMaxima);
		MI=miObs;

		jlDistLin=new JLabel("Dist ??.???");
		Font Grande = jlDistLin.getFont().deriveFont(20.0f);
		jlDistLin.setFont(Grande);
		jlDistLin.setHorizontalAlignment(JLabel.CENTER);
		jlDistLin.setEnabled(false);
		jpChecks.add(Box.createHorizontalStrut(15));
		jpChecks.add(jlDistLin);

		jlDistCamino=new JLabel("Camino ??.???");
		jlDistCamino.setFont(Grande);
		jlDistCamino.setHorizontalAlignment(JLabel.CENTER);
		jlDistCamino.setEnabled(false);
		jpChecks.add(Box.createHorizontalStrut(15));
		jpChecks.add(jlDistCamino);
		

	}

	protected void cosasAPintar(Graphics g0) {
		super.cosasAPintar(g0);
		Graphics2D g=(Graphics2D)g0;
		Point2D.Double pxCentro=point2Pixel(0.0,0.0);
		if(MI!=null && MI.posActual!=null) {
			//pintamos los bordes del camino
			Point2D.Double pxB=null;
			{//lado derecho
				GeneralPath gpBd=new GeneralPath();
				pxB=pointReal2pixel(MI.Bd[MI.iptoDini]);
				gpBd.moveTo((float)pxB.getX(), (float)pxB.getY());
				for(int i=MI.iptoDini+1; i<MI.Bd.length 
				&& MiraObstaculo.distanciaPuntos(MI.Bd[i],MI.posActual)<distanciaVista; i++) {
					pxB=pointReal2pixel(MI.Bd[i]);
					gpBd.lineTo((float)pxB.getX(), (float)pxB.getY());
				}
				g.setColor(Color.BLUE);
				g.draw(gpBd);
			}
			{//lado izquierdo
				GeneralPath gpBi=new GeneralPath();
				pxB=pointReal2pixel(MI.Bi[MI.iptoIini]);
				gpBi.moveTo((float)pxB.getX(), (float)pxB.getY());
				for(int i=MI.iptoIini+1; i<MI.Bi.length 
				&& MiraObstaculo.distanciaPuntos(MI.Bi[i],MI.posActual)<distanciaVista; i++) {
					pxB=pointReal2pixel(MI.Bi[i]);
					gpBi.lineTo((float)pxB.getX(), (float)pxB.getY());
				}
				g.setColor(Color.RED);
				g.draw(gpBi);
			}
			{// trayectoria
				GeneralPath gpTr=new GeneralPath();
				pxB=pointReal2pixel(MI.Tr[MI.indiceCoche]);
				gpTr.moveTo((float)pxB.getX(), (float)pxB.getY());
				for(int i=MI.indiceCoche+1; i<MI.Tr.length 
				&& MiraObstaculo.distanciaPuntos(MI.Tr[i],MI.posActual)<distanciaVista; i++) {
					pxB=pointReal2pixel(MI.Tr[i]);
					gpTr.lineTo((float)pxB.getX(), (float)pxB.getY());
				}
				g.setColor(Color.YELLOW);
				g.draw(gpTr);
			}
			//pintamos la distancia mínima etc.
			if(!java.lang.Double.isNaN(MI.dist))  {
				g.setStroke(new BasicStroke(2));
				g.setColor(Color.WHITE);
				//los de la derecha e izquierda que están libres
				g.draw(pathArrayXY(MI.Bd, MI.iptoDini, MI.iptoD+1));
				g.draw(pathArrayXY(MI.Bi, MI.iptoIini, MI.iptoI+1));

				if(MI.dist>0) {
					//marcamos el pto mínimo
					g.setStroke(new BasicStroke());
					g.setColor(Color.RED);
					g.draw(new Line2D.Double(pointReal2pixel(MI.posActual)
							,point2Pixel(ptoRF2Point(MI.indMin))));

					if(MI.iAD<MI.iAI) {
						g.setStroke(new BasicStroke(3));
						g.setColor(Color.RED);
						//pintamos rango de puntos en camino
						GeneralPath perimetro = 
							new GeneralPath(GeneralPath.WIND_EVEN_ODD, MI.iAI-MI.iAD+1);

						Point2D.Double px=point2Pixel(ptoRF2Point(MI.iAD));
						perimetro.moveTo((float)px.getX(),(float)px.getY());
						for(int i=MI.iAD+1; i<=MI.iAI; i++ ) {
							px=point2Pixel(ptoRF2Point(i));
							perimetro.lineTo((float)px.getX(),(float)px.getY());
						}
						g.draw(perimetro);
					}
				} else {
					//tenemos libre marcamos punto libre
					g.setStroke(new BasicStroke());
					g.setColor(Color.GREEN);
					g.draw(new Line2D.Double(pxCentro
							,pointReal2pixel(MI.Tr[MI.iLibre])));
				}
				g.setStroke(new BasicStroke());
				g.setColor(Color.GRAY);
				
				g.draw(new Line2D.Double(pxCentro
						,pointReal2pixel(MI.Bd[MI.iptoDini])));
				g.draw(new Line2D.Double(pxCentro
						,pointReal2pixel(MI.Bd[MI.iptoD])));
				g.draw(new Line2D.Double(pxCentro
						,pointReal2pixel(MI.Bi[MI.iptoIini])));
				g.draw(new Line2D.Double(pxCentro
						,pointReal2pixel(MI.Bi[MI.iptoI])));

				//Pintamos en verde la distancia sobre el camino
				if(!Double.isInfinite(MI.distCamino) && MI.indSegObs<MI.Tr.length) {
					//tenemos los índices
					g.setStroke(new BasicStroke(3));
					g.setColor(Color.GREEN);
					GeneralPath gp=pathArrayXY(MI.Tr, MI.indiceCoche
							, MI.indSegObs+1);
					if(gp!=null)
						g.draw(gp);
					g.draw(new Line2D.Double(pointReal2pixel(MI.Bi[MI.indSegObs])
							,pointReal2pixel(MI.Bd[MI.indSegObs])));
					g.draw(new Line2D.Double(pointReal2pixel(MI.Bi[MI.indiceCoche])
							,pointReal2pixel(MI.Bd[MI.indiceCoche])));
					if(MI.indBarrSegObs!=Integer.MAX_VALUE) {
						//marcamos pto barrido dió obstáculo camino más cercano
						g.setStroke(new BasicStroke());
						g.draw(new Line2D.Double(pxCentro
								,point2Pixel(ptoRF2Point(MI.indBarrSegObs))));
					}
				}

			}
		}

	}


	/**
	 * Genera {@link GeneralPath} con puntos en array expresados en sistema Real
	 * @param v array de al menos 2 columnas. La primera se considera coordenada X, la segunda la Y
	 * @param iini indice del primer punto
	 * @param ifin indice siguiente del último punto
	 * @return {@link GeneralPath} con los puntos considerados
	 */
	protected GeneralPath pathArrayXY(double [][] v, int iini, int ifin) {
		if(iini<0 || ifin<iini || v==null || v.length<ifin || v[0].length<2)
			return null;
		GeneralPath perimetro = 
			new GeneralPath(GeneralPath.WIND_EVEN_ODD, ifin-iini);

		Point2D.Double px=pointReal2pixel(v[iini][0],v[iini][1]);
		perimetro.moveTo((float)px.getX(),(float)px.getY());
		for(int i=iini+1; i<ifin; i++) {
			px=pointReal2pixel(v[i][0],v[i][1]);
			//Siguientes puntos son lineas
			perimetro.lineTo((float)px.getX(),(float)px.getY());
		}
		return perimetro;
	}

	
	/**
	 * Para cambiar el barrido que se está mostrando.
	 * y actualiza la presentación
	 */
	public void actualiza() {
		if(MI==null)
			return;
		if(java.lang.Double.isNaN(MI.dist)) {
			jlDistLin.setText("Fuera");
			jlDistLin.setForeground(Color.RED);
		} else 
			if (MI.dist>0) {
				jlDistLin.setText(String.format("Lineal %6.3f m", MI.dist));
				jlDistLin.setForeground(Color.RED);
			} else {
				jlDistLin.setText(String.format("Lineal %6.3f m", -MI.dist));
				jlDistLin.setForeground(Color.GREEN);
			}
		jlDistLin.setEnabled(true);

		if(!Double.isInfinite(MI.distCamino)) {
			jlDistCamino.setText(String.format("Camino %6.3f m", MI.distCamino));
			jlDistCamino.setEnabled(true);
		}
		
		setBarrido(MI.barr);
		repaint();
	}


	/**
	 * @param args
	 */
	public static void main(String[] args) {

		double[][] Tr={{ 0.000000 , 0.000000}
		,{ 0.049769 , 0.707045}
		,{ 0.087096 , 1.237328}
		,{ 0.136865 , 1.944373}
		,{ 0.237998 , 2.446245}
		,{ 0.287767 , 3.153290}
		,{ 0.325094 , 3.683574}
		,{ 0.374863 , 4.390618}
		,{ 0.412190 , 4.920902}
		,{ 0.461959 , 5.627947}
		,{ 0.499286 , 6.158230}
		,{ 0.549055 , 6.865275}
		,{ 0.650187 , 7.367148}
		,{ 0.699956 , 8.074193}
		,{ 0.673477 , 8.632888}
		,{ 0.723246 , 9.339933}
		,{ 0.598399 , 9.837631}
		,{ 0.648168 , 10.544676}
		,{ 0.523321 , 11.042374}
		,{ 0.398474 , 11.540072}
		,{ 0.111454 , 12.005184}
		,{ -0.175567 , 12.470297}
		,{ -0.312856 , 12.791234}
		,{ -0.774493 , 13.046999}
		,{ -1.108518 , 13.245943}
		,{ -1.518791 , 13.296536}
		,{ -1.992870 , 13.375540}
		,{ -2.641565 , 13.245197}
		,{ -3.064280 , 13.119029}
		,{ -3.875149 , 12.956100}
		,{ -4.297864 , 12.829932}
		,{ -4.946560 , 12.699590}
		,{ -5.595255 , 12.569247}
		,{ -6.180145 , 12.410493}
		,{ -6.666666 , 12.312736}
		,{ -7.489978 , 11.973047}
		,{ -7.912694 , 11.846879}
		,{ -8.659757 , 11.655539}
		,{ -9.146279 , 11.557782}
		,{ -9.731169 , 11.399029}
		,{ -10.153886 , 11.272861}
		,{ -10.900950 , 11.081522}
		,{ -11.323666 , 10.955354}
		,{ -12.070731 , 10.764015}
		,{ -12.569695 , 10.489497}
		,{ -13.218391 , 10.359154}
		,{ -13.867088 , 10.228812}
		,{ -14.451978 , 10.070059}
		,{ -14.938501 , 9.972302}
		,{ -15.749371 , 9.809374}
		,{ -16.172088 , 9.683207}
		,{ -16.982959 , 9.520279}
		,{ -17.469482 , 9.422522}
		,{ -18.216547 , 9.231184}
		,{ -18.703070 , 9.133427}
		,{ -19.513941 , 8.970500}
		,{ -20.000464 , 8.872743}
		,{ -20.585356 , 8.713990}
		,{ -21.234053 , 8.583648}
		,{ -21.882750 , 8.453306}
		,{ -22.519005 , 8.499726}
		,{ -23.155260 , 8.546146}
		,{ -23.565536 , 8.596741}
		,{ -24.189349 , 8.819923}
		,{ -24.663430 , 8.898928}
		,{ -25.112626 , 9.331458}
		,{ -25.561823 , 9.763987}
		,{ -25.836403 , 10.405864}
		,{ -26.059620 , 10.842568}
		,{ -26.159583 , 11.693793}
		,{ -26.284431 , 12.191493}
		,{ -26.320589 , 13.014307}
		,{ -26.283262 , 13.544593}
		,{ -26.221051 , 14.428403}
		,{ -26.072913 , 15.196447}
		,{ -25.848528 , 16.112843}
		,{ -25.811201 , 16.643129}
		,{ -25.424641 , 17.592110}
		,{ -25.237583 , 17.978219}
		,{ -24.851023 , 18.927199}
		,{ -24.715327 , 19.518482}
		,{ -24.255283 , 20.174934}
		,{ -24.055782 , 20.737806}
		,{ -23.595738 , 21.394258}
		,{ -23.234063 , 21.989715}
		,{ -22.774019 , 22.646167}
		,{ -22.262612 , 23.097447}
		,{ -21.751204 , 23.548726}
		,{ -21.252239 , 23.823244}
		,{ -20.578658 , 24.307109}
		,{ -20.079693 , 24.581626}
		,{ -19.418553 , 24.888729}
		,{ -18.932031 , 24.986485}
		,{ -18.184965 , 25.177822}
		,{ -17.686000 , 25.452339}
		,{ -16.875129 , 25.615265}
		,{ -16.401049 , 25.536259}
		,{ -15.590178 , 25.699185}
		,{ -14.941481 , 25.829526}
		,{ -14.194416 , 26.020864}
		,{ -13.707894 , 26.118619}
		,{ -12.960829 , 26.309957}
		,{ -12.375939 , 26.468708}
		,{ -11.565068 , 26.631635}
		,{ -10.980178 , 26.790387}
		,{ -10.233114 , 26.981724}
		,{ -9.746592 , 27.079480}
		,{ -8.837354 , 27.303403}
		,{ -8.414638 , 27.429570}
		,{ -7.441594 , 27.625082}
		,{ -7.018878 , 27.751249}
		,{ -6.045835 , 27.946761}
		,{ -5.559314 , 28.044517}
		,{ -4.812250 , 28.235855}
		,{ -4.325729 , 28.333611}
		,{ -3.578666 , 28.524948}
		,{ -2.929970 , 28.655290}
		,{ -2.182908 , 28.846628}
		,{ -1.696386 , 28.944384}
		,{ -1.175303 , 29.131547}
		,{ -0.590414 , 29.290300}
		,{ -0.005525 , 29.449052}
		,{ 0.655611 , 29.756156}
		,{ 1.240500 , 29.914908}
		,{ 1.889194 , 30.045250}
		,{ 2.537889 , 30.175592}
		,{ 3.024410 , 30.273348}
		,{ 3.707665 , 30.493097}
		,{ 4.194186 , 30.590854}
		,{ 4.842880 , 30.721196}
		,{ 5.265595 , 30.847363}
		,{ 6.140268 , 30.981880}
		,{ 6.626789 , 31.079636}
		,{ 7.373850 , 31.270975}
		,{ 7.796564 , 31.397142}
		,{ 8.607431 , 31.560070}
		,{ 9.192319 , 31.718823}
		,{ 9.939380 , 31.910161}
		,{ 10.575631 , 31.863742}
		,{ 11.386498 , 32.026670}
		,{ 11.971385 , 32.185423}
		,{ 12.629755 , 32.051650}
		,{ 12.814047 , 31.996881}
		,{ 13.361608 , 31.625351}
		,{ 13.521016 , 31.217060}
		,{ 13.620978 , 30.365840}
		,{ 13.971805 , 29.872317}
		,{ 13.973400 , 28.960100}
		,{ 13.987438 , 28.224645}
		,{ 13.976592 , 27.135667}
		,{ 13.828456 , 26.367626}
		,{ 13.667878 , 25.422824}
		,{ 13.618110 , 24.715780}
		,{ 13.607263 , 23.626802}
		,{ 13.621301 , 22.891347}
		,{ 13.598012 , 21.625608}
		,{ 13.484437 , 20.946975}
		,{ 13.247611 , 19.853822}
		,{ 13.185400 , 18.970017}
		,{ 13.098305 , 17.732690}
		,{ 13.048536 , 17.025645}
		,{ 13.025247 , 15.759907}
		,{ 12.911672 , 15.081274}
		,{ 12.888382 , 13.815536}
		,{ 12.902420 , 13.080081}
		,{ 12.879130 , 11.814343}
		,{ 12.893168 , 11.078888}
		,{ 12.707705 , 9.780564}
		,{ 12.657936 , 9.073520}
		,{ 12.519476 , 8.041365}
		,{ 12.405901 , 7.362732}
		,{ 12.220438 , 6.064409}
		,{ 12.234475 , 5.328954}
		,{ 12.159822 , 4.268388}
		,{ 12.110052 , 3.561345}
		,{ 12.022956 , 2.324018}
		,{ 11.973187 , 1.616975}
		,{ 11.898534 , 0.556409}
		,{ 11.622785 , -0.154809}
		,{ 11.535689 , -1.392135}
		,{ 11.485920 , -2.099179}
		,{ 11.185286 , -3.163919}
		,{ 11.123075 , -4.047723}
		,{ 10.984614 , -5.079877}
		,{ 10.934845 , -5.786920}
		,{ 10.621769 , -7.028421}
		,{ 10.508194 , -7.707053}
		,{ 10.421097 , -8.944378}
		,{ 10.371328 , -9.651421}
		,{ 10.070694 , -10.716161}
		,{ 10.072288 , -11.628376}
		,{ 9.997634 , -12.688941}
		,{ 9.947864 , -13.395983}
		,{ 9.860768 , -14.633309}
		,{ 9.747192 , -15.311940}
		,{ 9.660095 , -16.549265}
		,{ 9.610325 , -17.256308}
		,{ 9.361055 , -18.526219}
		,{ 9.298843 , -19.410022}
		,{ 9.373919 , -20.614760}
		,{ 9.098170 , -21.325978}
		,{ 9.011073 , -22.563302}
		,{ 8.961303 , -23.270345}
		,{ 8.874206 , -24.507669}
		,{ 8.748187 , -25.363060}
		,{ 8.673532 , -26.423624}
		,{ 8.547514 , -27.279015}
		,{ 8.460417 , -28.516339}
		,{ 8.398204 , -29.400141}
		,{ 8.161376 , -30.493291}
		,{ 8.035357 , -31.348682}
		,{ 7.862335 , -32.470244}
		,{ 7.650391 , -33.209873}
		,{ 7.187583 , -34.307198}
		,{ 6.988082 , -34.870066}
		,{ 6.439349 , -35.851629}
		,{ 6.077675 , -36.447084}
		,{ 5.443017 , -37.312885}
		,{ 5.007860 , -37.615818}
		,{ 4.172108 , -38.132273}
		,{ 3.673145 , -38.406794}
		,{ 3.075814 , -38.742313}
		,{ 2.589294 , -38.840074}
		,{ 1.842232 , -39.031419}
		,{ 1.355711 , -39.129180}
		,{ 0.719460 , -39.082766}
		,{ 0.245382 , -39.003766}
		,{ -0.314622 , -38.809004}
		,{ -0.712451 , -38.581655}
		,{ -1.260012 , -38.210131}
		,{ -1.547032 , -37.745023}
		,{ -2.082150 , -37.196739}
		,{ -2.369170 , -36.731630}
		,{ -2.618862 , -35.736239}
		,{ -2.731266 , -35.061783}
		,{ -2.592805 , -34.029630}
		,{ -2.606840 , -33.294175}
		,{ -2.532185 , -32.233610}
		,{ -2.482415 , -31.526567}
		,{ -2.395317 , -30.289242}
		,{ -2.345547 , -29.582199}
		,{ -2.270892 , -28.521634}
		,{ -2.208680 , -27.637830}
		,{ -2.134025 , -26.577265}
		,{ -2.008007 , -25.721872}
		,{ -1.933352 , -24.661307}
		,{ -1.883582 , -23.954263}
		,{ -1.796485 , -22.716937}
		,{ -1.746715 , -22.009894}
		,{ -1.659618 , -20.772567}
		,{ -1.546042 , -20.093935}
		,{ -1.535193 , -19.004958}
		,{ -1.472981 , -18.121154}
		,{ -1.334521 , -17.088999}
		,{ -1.284751 , -16.381956}
		,{ -1.133848 , -15.173040}
		,{ -1.084079 , -14.465996}
		,{ -0.933176 , -13.257081}
		,{ -0.883407 , -12.550037}
		,{ -0.796310 , -11.312710}
		,{ -0.746540 , -10.605665}
		,{ -0.659444 , -9.368338}
		,{ -0.609674 , -8.661294}
		,{ -0.586384 , -7.395555}
		,{ -0.536614 , -6.688511}
		,{ -0.287344 , -5.418597}
		,{ -0.237575 , -4.711553}
		,{ -0.150479 , -3.474225}
		};
		
		System.out.println("Longitud de la trayectoria="+Tr.length);
		
		MiraObstaculo mi=new MiraObstaculo(Tr);		
		PanelMiraObstaculoSubjetivo pMOS=new PanelMiraObstaculoSubjetivo(mi,(short)80);
		

		JFrame VentanaPrincipal=new JFrame("PanelMuestraBarridoSubjetivo");
		VentanaPrincipal.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		VentanaPrincipal.add(pMOS);
		VentanaPrincipal.setSize(new Dimension(800,400));
		VentanaPrincipal.setVisible(true);
		
		JFrame ventana=new JFrame("Panel Mira Obstáculo");
		ventana.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		PanelMiraObstaculo pmo=new PanelMiraObstaculo(mi);
		ventana.add(pmo);
		ventana.setSize(new Dimension(800,600));
		ventana.setVisible(true);

		
		pMOS.setZona(new ZonaRadialLMS((short)180,(short)50,true,true,ZonaLMS.ZONA_A,(short)25000));
		
		pMOS.setZona(new ZonaRectangularLMS((short)180,(short)50,true,true,ZonaLMS.ZONA_B
				,(short)20000,(short)15000,(short)30000));

		ZonaSegmentadaLMS zs=new ZonaSegmentadaLMS((short)180,(short)100,false,true,ZonaLMS.ZONA_C,(short)30);
		for(int i=0; i<=30; i++)
			zs.radiosPuntos[i]=990;
		zs.radiosPuntos[0]=0;
		zs.radiosPuntos[30]=0;
		pMOS.setZona(zs);
		
		//Damos pto, orientación y barrido
		BarridoAngular ba=new BarridoAngular(181,0,4,(byte)2,false,(short)2);
		for(int i=0;i<ba.numDatos();i++) {
//			ba.datos[i]=(short)((15.0)*100.0);
			ba.datos[i]=(short)((Math.sin((double)i/(ba.numDatos()-1)*Math.PI*13.6)*3.0+10.0)*100.0);
		}
		double[] ptoAct={-26, 14};
		double dist=mi.masCercano(ptoAct, Math.toRadians(90), ba);
		pmo.actualiza();
		pMOS.actualiza();
		System.out.println("Distancia="+dist);
		System.out.println(" iAD="+pMOS.MI.iAD
				+"\n iAI="+pMOS.MI.iAI
				+"\n iptoD ="+pMOS.MI.iptoD
				+" \n iptoI ="+pMOS.MI.iptoI
				+" \n iptoDini ="+pMOS.MI.iptoDini
				+" \n iptoIini ="+pMOS.MI.iptoIini
				+" \n imin ="+pMOS.MI.indMin
				);

		boolean Caminar=true;
		if(Caminar) {
			//esparamos antes de empezar a caminar
			try {
				Thread.sleep(5000);
			} catch (Exception e) { }
			//vamos recorriendo la trayectoria con barridos aleatorios
			int inTr=20, inTrAnt=inTr-2;
			while(true) {
				BarridoAngular barAct=new BarridoAngular(181,0,4,(byte)2,false,(short)2);
				double frec=(13.6+2*Math.random());
				double Amp=(3.0+15*Math.random());
				double Dpor=(20.0+15*Math.random());
				for(int i=0;i<barAct.numDatos();i++) {
//					barAct.datos[i]=(short)((15.0)*100.0);
					barAct.datos[i]=(short)((Math.sin((double)i/(barAct.numDatos()-1)*Math.PI*frec)
							*Amp
							+Dpor)*100.0);
				}

				double diAct=mi.masCercano(Tr[inTr]
				                               , Math.atan2(Tr[inTr][1]-Tr[inTrAnt][1],Tr[inTr][0]-Tr[inTrAnt][0]), barAct);
				pmo.actualiza();
				pMOS.actualiza();
				System.out.println("Indice "+inTr+" distancia "+diAct);
				try {
					Thread.sleep(3000);
				} catch (Exception e) { }
				inTrAnt=inTr;
				inTr=(inTr+1)%Tr.length;
			}
		}

		
	}


}
