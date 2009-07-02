/**
 * 
 */
package sibtra.util;

/**
 * @author alberto
 *
 */
public class PanelJoystick extends PanelDatos {
	
	private ManejaJoystick manejaJoystick;

	public PanelJoystick(ManejaJoystick manJoy) {
		super();
		manejaJoystick=manJoy;
		
		añadeAPanel(new LabelDatoFormato("#######",ManejaJoystick.class,"getX","%7f"), "X");
		añadeAPanel(new LabelDatoFormato("#######",ManejaJoystick.class,"getY","%7f"), "Y");
		añadeAPanel(new LabelDatoFormato("###.##",ManejaJoystick.class,"getAlfaGrados","%6.2f º"), "Alfa");
		añadeAPanel(new LabelDatoFormato("#.###",ManejaJoystick.class,"getVelocidad","%5.3f m/s"), "Velocidad");
		añadeAPanel(new LabelDatoFormato("####",ManejaJoystick.class,"getAvance","%+3.0f"), "Avance");
		
	}
	
	public void actualiza() {
		manejaJoystick.poll();
		actualizaDatos(manejaJoystick);
		super.actualiza();
	}
	

}
