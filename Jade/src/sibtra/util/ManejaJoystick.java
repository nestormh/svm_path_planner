package sibtra.util;

import java.io.IOException;

import com.centralnexus.input.Joystick;
/**
 * Calse para manejar el Joytick de volante y pedales.
 * Usa valores de calibarción por defecto.
 * @author alberto
 *
 */
public class ManejaJoystick {
	
	//TODO utilizar punto central y extermos para los cáculos para considerar zona muerta central y dar mejor 0	

	public final static float MinY=0.0014648885f;
	public final static float MaxY=0.005401776f;
	public final static float MinX=5.1881466E-4f;
	public final static float MaxX=0.0055543687f;
    double mitadY=(MaxY+MinY)/2;

	double AlfaMaximo=Math.toRadians(45);
	double VelocidaMaxima=6;
	double AvanceMaximo=255;
//	double FrenoMaximo=-255;
	
	private Joystick joystick;
	
	public ManejaJoystick() {
		try {
			joystick=Joystick.createInstance();
		} catch (IOException exp) {
			System.err.println("No ha sido posible conectar al joystick");
			joystick=null;
		}
	}
	
	public double getAlfa() {
        float x=joystick.getX();
        double alfa=-(AlfaMaximo*2/(MaxX-MinX)*(x-MinX)-AlfaMaximo);
        return alfa;
	}

	/** Se usará el acelerador como consigna de velocidad.
	 * Por ello si se pisa el freno la consigna será 0.
	 * @return
	 */
	public double getVelocidad() {
		//el eje y está invertido y pequeño acelerar, y grade frenar
        float y=joystick.getY();
        //si está frenanado
        if(y>mitadY) return 0.0;
		double velMS=VelocidaMaxima/(mitadY-MinY)*(mitadY-y);
		return velMS;
	}
	
	public double getAvance() {
		//el eje y está invertido y pequeño acelerar, y grade frenar
        float y=joystick.getY();
        double avance=(2*AvanceMaximo)/(MinY-MaxY)*(y-MinY)+AvanceMaximo;
        return avance;
	}

	public void poll() {
		joystick.poll();
	}
	
	public double getX() {
		return joystick.getX();
	}
	
	public double getY() {
		return joystick.getY();
	}
}
