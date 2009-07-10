package sibtra.util;

import java.io.IOException;

import com.centralnexus.input.Joystick;
/**
 * Clase para manejar el Joytick de volante y pedales.
 * Usa valores de calibarción por defecto.
 * Usa el primer joystick disponible.
 * Hace ajuste lineal usando valores minimos y máximos en cada eje, se debería mejorar para considerar zona muerta.
 * @author alberto
 *
 */
public class ManejaJoystick {
	
	//TODO utilizar punto central y extermos para los cáculos para considerar zona muerta central y dar mejor 0	

	float MinY=0.0014648885f;
	float MaxY=0.005401776f;
	float MinX=5.1881466E-4f;
	float MaxX=0.0055543687f;
    double mitadY=(MaxY+MinY)/2;

    /** Ángulo máximo para el volante */
	double AlfaMaximo=Math.toRadians(45);
	/** Velocidad máxima para consigna */
	double VelocidaMaxima=6;
	/** Avance máximo en positivo y negativo */
	double AvanceMaximo=255;
	
	private Joystick joystick;
	
	/** Usará el primer joystick disponible */
	public ManejaJoystick() {
		try {
			joystick=Joystick.createInstance();
		} catch (IOException exp) {
			System.err.println("No ha sido posible conectar al joystick");
			joystick=null;
		}
	}
	
	/** @return Valor de ángulo del volante, NaN si no hay joystick */
	public double getAlfa() {
		if(joystick==null) return Double.NaN;
        float x=joystick.getX();
        if(x>MaxX) return AlfaMaximo;
        if(x<MinX) return -AlfaMaximo;
        double alfa=-(AlfaMaximo*2/(MaxX-MinX)*(x-MinX)-AlfaMaximo);
        return alfa;
	}

	/** @return valor del ángulo del volante en grados */
	public double getAlfaGrados() {
        return Math.toDegrees(getAlfa());
	}

	/** Se usará el acelerador como consigna de velocidad.
	 * Por ello si se pisa el freno la consigna será 0.
	 * @return consinga del velocidad o NaN si no está definido {@link #joystick}
	 */
	public double getVelocidad() {
		if(joystick==null) return Double.NaN;
		//el eje y está invertido y pequeño acelerar, y grade frenar
        float y=joystick.getY();
        //si está frenanado
        if(y>mitadY) return 0.0;
        if(y<MinY) return VelocidaMaxima;
		double velMS=VelocidaMaxima/(mitadY-MinY)*(mitadY-y);
		return velMS;
	}
	
	/** @return el avance o NaN si {@link #joystick} no está definido */
	public double getAvance() {
		if(joystick==null) return Double.NaN;
		//el eje y está invertido y pequeño acelerar, y grade frenar
        float y=joystick.getY();
        if(y>MaxY) return -AvanceMaximo;
        if(y<MinY) return AvanceMaximo;
        double avance=(2*AvanceMaximo)/(MinY-MaxY)*(y-MinY)+AvanceMaximo;
        return avance;
	}

	/** Actuliza los valores del joystick */
	public void poll() {
		if(joystick==null) return;
		joystick.poll();
	}
	
	/** @return getX() del joystick */
	public double getX() {
		if(joystick==null) return Double.NaN;
		return joystick.getX();
	}
	
	/** @return getY() del joystick */
	public double getY() {
		if(joystick==null) return Double.NaN;
		return joystick.getY();
	}
	
	/** @return si se ha conseguido el joystick */
	public boolean hayJoystick() {
		return joystick!=null;
	}
}
